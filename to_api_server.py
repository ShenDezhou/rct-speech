import argparse
import logging
import os
from types import SimpleNamespace
import falcon
from falcon_cors import CORS
import json
import waitress
import time


import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from arguments import get_args
from utils import Timers
from utils import load_checkpoint_model
from data_utils.tokenization_gpt2 import GPT2Tokenizer
from configure_data import configure_data
import mpu

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0

USE_TORCH_DDP = False

logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
logger = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['*'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-p', '--port', default=58100,
#     help='falcon server port')
# parser.add_argument(
#     '-c', '--config_file', default='config/config.json',
#     help='model config file')
# args = parser.parse_args()
port = os.getenv('TO_PORT', 8010)
device_affinity = os.getenv('DEVICE_AFFINITY', 0)
# model_config= 'config/config.json'
torch.cuda.set_device(int(device_affinity))

class TorchResource:

    def __init__(self):
        logger.info("...")
        # Arguments.
        self.args = get_args()
        # 0. Load config
        # with open(model_config) as fin:
        #     self.config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

        self.initialize_distributed(self.args)

        # Random seeds for reproducability.
        self.set_random_seed(self.args.seed)

        # get the tokenizer
        self.tokenizer = GPT2Tokenizer(os.path.join(self.args.tokenizer_path, 'vocab.json'),
                                  os.path.join(self.args.tokenizer_path, 'chinese_vocab.model'))

        self.args.parallel_output = False
        self.model = self.setup_model(self.args)
        self.args.batch_size = 1

        logger.info("###")

    def initialize_distributed(self, args):
        """Initialize torch.distributed."""

        # Manually set the device ids.
        device = args.rank % torch.cuda.device_count()
        if args.local_rank is not None:
            device = args.local_rank
        if device_affinity:
            device = int(device_affinity)
        print('device:',device)
        self.device = device
        torch.cuda.set_device(device)
        # Call the init process
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        # init_method += master_ip + ':' + master_port
        init_method += master_ip + ':' + '12580'[:-1] + str(device_affinity)
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)

        # Set the model-parallel / data-parallel communicators.
        mpu.initialize_model_parallel(args.model_parallel_size)

    def set_random_seed(self, seed):
        """Set random seed for reproducability."""

        if seed is not None and seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            mpu.model_parallel_cuda_manual_seed(seed)

    def get_model(self, args):
        """Build the model."""

        print_rank_0('building CPM model ...')
        self.model = GPT2Model(num_layers=args.num_layers,
                          vocab_size=args.vocab_size,
                          hidden_size=args.hidden_size,
                          num_attention_heads=args.num_attention_heads,
                          embedding_dropout_prob=args.hidden_dropout,
                          attention_dropout_prob=args.attention_dropout,
                          output_dropout_prob=args.hidden_dropout,
                          max_sequence_length=args.max_position_embeddings,
                          checkpoint_activations=args.checkpoint_activations,
                          checkpoint_num_layers=args.checkpoint_num_layers,
                          parallel_output=args.parallel_output)

        if mpu.get_data_parallel_rank() == 0:
            print(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in self.model.parameters()])), flush=True)

        # GPU allocation.
        self.model.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if args.fp16:
            self.model = FP16_Module(self.model)

        # Wrap model for distributed training.
        if USE_TORCH_DDP:
            i = torch.cuda.current_device()
            self.model = DDP(self.model, device_ids=[i], output_device=i,
                        process_group=mpu.get_data_parallel_group())
        else:
            self.model = DDP(self.model)

        return self.model

    def setup_model(self, args):
        """Setup model."""

        self.model = self.get_model(args)

        args.iteration = load_checkpoint_model(self.model, args)

        return self.model


    def get_masks_and_position_ids(self, data,
                                   eod_token,
                                   reset_position_ids,
                                   reset_attention_mask):
        # Extract batch size and sequence length.
        batch_size, seq_length = data.size()

        # Attention mask (lower triangular).
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

        # Loss mask.
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
        loss_mask[data == eod_token] = 0.0

        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

        return attention_mask, loss_mask, position_ids


    def get_batch(self, context_tokens, device, args):
        tokens = context_tokens
        tokens = tokens.view(args.batch_size, -1).contiguous()
        tokens = tokens.to(device)

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = self.get_masks_and_position_ids(
            tokens,
            args.eod_token,
            args.reset_position_ids,
            args.reset_attention_mask)

        return tokens, attention_mask, position_ids

    def top_k_logits(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        # This function has been mostly taken from huggingface conversational ai code at
        # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # convert to 1D
            logits = logits.view(logits.size()[1]).contiguous()
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
            # going back to 2D
            logits = logits.view(1, -1).contiguous()

        return logits

    def generate_samples(self, config, model, tokenizer, args, device):  # 产出文本
        raw_text = config['prompt']
        output_len = config.get('length', args.out_seq_length)
        top_k = config.get('top_k', args.top_k)
        top_p = config.get('top_p', args.top_p)
        temperature = config.get('temperature', args.temperature)
        print_rank_0(raw_text)
        context_count = 0
        model.eval()
        with torch.no_grad():  # 关闭自动求导引擎
            # while True:  # 当循环不中止的时候一直进行下去
            #     torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            # <操作1>在此处直接写规则（函数另其对对话进行直接返回）
            if mpu.get_model_parallel_rank() == 0:
                # if args.input_text:  # 如果满足，则raw_text取example.txt
                #     # 交互式文本
                #     raw_text = open(args.input_text).read().strip()
                # else:
                #     raw_text = input("\nContext prompt (stop to exit) >>> ")
                #     while not raw_text:
                #         print('Prompt should not be empty!')
                #         raw_text = input("\nContext prompt (stop to exit) >>> ")

                # if "stop" in raw_text:
                #     terminate_runs = 1  # 终止信号，为1则终止；为0则继续
                # else:
                # context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
                context_tokens = tokenizer.encode(raw_text)  # 得到了经过分词后的list:token_id  [t1,t2,t3,...,tn]
                context_length = len(context_tokens)  # 获取这些tokenid list的长度

                if context_length >= args.seq_length // 2:
                    print("\nContext length ", context_length, \
                          "\nPlease give smaller context (half of the sequence length)!")
                    return "\nContext length " + str(context_length) + \
                          "\nPlease give smaller context (half of the sequence length)!"
                    # continue  # 重新输入
            else:
                # context_tokens = tokenizer.EncodeAsIds("EMPTY TEXT").tokenization
                context_tokens = tokenizer.encode("空文本")
                context_length = len(context_tokens)
            # print_rank_0(context_tokens)
            # terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            # torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
            #                             group=mpu.get_model_parallel_group())
            # terminate_runs = terminate_runs_tensor[0].item()

            # if terminate_runs == 1:
            #     return
            # <操作2>保存中间变量，并开始写规则和循环让模型预测进行回溯继续预测(可以找一下topk的排序预测问题)：
            pad_id = tokenizer.encoder['<pad>']
            args.eod_token = tokenizer.encoder['<eod>']
            if context_length < args.seq_length:
                context_tokens.extend([pad_id] * (args.seq_length - context_length))

            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            context_length_tensor = torch.cuda.LongTensor([context_length])
            # print_rank_0(context_tokens_tensor)

            # torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
            #                             group=mpu.get_model_parallel_group())
            # torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
            #                             group=mpu.get_model_parallel_group())
            # print_rank_0(context_tokens_tensor)
            context_length = context_length_tensor[0].item()
            tokens, attention_mask, position_ids = self.get_batch(context_tokens_tensor, device, args)

            start_time = time.time()

            counter = 0
            org_context_length = context_length

            past_key_values = None
            while counter < (org_context_length + output_len):  # 使用GPT2进行反复生成式得到回答？<可以在中间测一下>
                # print('test')
                if counter == 0:
                    # print_rank_0(tokens[:, :context_length])
                    logits, past_key_values = model(tokens[:, :context_length], position_ids[:, :context_length],
                                                    attention_mask[:, :, :context_length, :context_length],
                                                    past_key_values=past_key_values, use_cache=True)
                    logits = logits[:, context_length - 1, :]
                    counter += context_length
                    # print_rank_0(logits)
                else:
                    # print_rank_0(tokens[:, context_length - 1: context_length])
                    logits, past_key_values = model(tokens[:, context_length - 1: context_length],
                                                    position_ids[:, context_length - 1: context_length],
                                                    attention_mask[:, :, context_length - 1, :context_length],
                                                    past_key_values=past_key_values, use_cache=True)
                    logits = logits[:, 0, :]
                    # print_rank_0(logits)

                past_key_values = [x.half() for x in past_key_values]
                logits = self.top_k_logits(logits, top_k=top_k, top_p=top_p)
                log_probs = F.softmax(logits / temperature, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                tokens[0, context_length] = prev[0]
                # print_rank_0(tokens[0, context_length])
                # torch.distributed.broadcast(tokens, mpu.get_model_parallel_src_rank(),
                #                             group=mpu.get_model_parallel_group())
                context_length += 1
                counter += 1

                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.decode(output_tokens_list.tolist())
                token_end = decode_tokens.find("<eod>")

                # if mpu.get_model_parallel_rank() == 0 and (counter % 16 == 0 or token_end != -1):
                #     # os.system('clear')
                #     print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                #     print("\nContext:", raw_text, flush=True)
                #     trim_decode_tokens = decode_tokens[len(raw_text):decode_tokens.find("<eod>")]
                #     print("\nCPM:", trim_decode_tokens, flush=True)

                if token_end != -1:
                    # print(token_end)
                    break

            if mpu.get_model_parallel_rank() == 0:
                # os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.decode(output_tokens_list.tolist())
                trim_decode_tokens = decode_tokens[len(raw_text):decode_tokens.find("<eod>")]
                print("\nCPM:", trim_decode_tokens, flush=True)

                # print(token_end)
            #raw_text = None

            # torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

            # if args.input_text:
            #     break
            return trim_decode_tokens

    def gpt_generate(self, jsondata):
        logger.info(jsondata)
        number = jsondata.get('number', 3)
        answer_list = []
        for _ in range(number):
            answer_list.append(self.generate_samples(jsondata, self.model, self.tokenizer, self.args, torch.cuda.current_device()))
        return {"result": answer_list}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        jsondata = req.get_param('json', True)
        # content = req.get_param('2', True)
        # clean_title = shortenlines(title)
        # clean_content = cleanall(content)
        resp.media = self.gpt_generate(jsondata)
        logger.info("###")


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        jsondata = json.loads(data)
        # clean_title = shortenlines(jsondata.title)
        # clean_content = cleanall(jsondata.content)
        torch.cuda.set_device(self.device)
        print('device:', self.device)
        resp.media = self.gpt_generate(jsondata)

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=port, threads=48, url_scheme='http')