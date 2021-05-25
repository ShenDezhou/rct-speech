import argparse
import io
import logging
import mimetypes
import os
import re
import uuid
from types import SimpleNamespace
import falcon
from falcon_cors import CORS
import json
import waitress
import torch.nn.functional as F
import argparse
import time

import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os


import hparams as hp
import audio
import utils
import dataset
import text
import model as M
import waveglow

USE_TORCH_DDP = False

logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
logger = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['*'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', type=int, default=8010)
parser.add_argument('--step', type=int, default=336000)
parser.add_argument("--alpha", type=float, default=1.0)
args = parser.parse_args()
# args = parser.parse_args()
# port = os.getenv('TO_PORT', 8010)
device_affinity = os.getenv('DEVICE_AFFINITY', 1)
# model_config= 'config/config.json'
torch.cuda.set_device(int(device_affinity))
# class ImageStore:
#
#     _CHUNK_SIZE_BYTES = 4096
#     _IMAGE_NAME_PATTERN = re.compile(
#         '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.[a-z]{2,4}$'
#     )
#
#     def __init__(self, storage_path, uuidgen=uuid.uuid4, fopen=io.open):
#         self._storage_path = storage_path
#         self._uuidgen = uuidgen
#         self._fopen = fopen
#
#     def save(self, image_stream, image_content_type):
#         ext = mimetypes.guess_extension(image_content_type)
#         name = '{uuid}{ext}'.format(uuid=self._uuidgen(), ext=ext)
#         image_path = os.path.join(self._storage_path, name)
#
#         with self._fopen(image_path, 'wb') as image_file:
#             while True:
#                 chunk = image_stream.read(self._CHUNK_SIZE_BYTES)
#                 if not chunk:
#                     break
#
#                 image_file.write(chunk)
#
#         return name
#
#     def open(self, name):
#         # Always validate untrusted input!
#         # if not self._IMAGE_NAME_PATTERN.match(name):
#         #     raise IOError('File not found')
#
#         image_path = os.path.join(self._storage_path, name)
#         stream = self._fopen(image_path, 'rb')
#         content_length = os.path.getsize(image_path)
#
#         return stream, content_length

class TorchResource:

    def __init__(self):
        logger.info("...")
        # Arguments.
        # self.store = ImageStore(storage_path='.')
        # Test
        self.WaveGlow = utils.get_WaveGlow()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("use griffin-lim and waveglow")
        self.model = self.get_DNN(args.step)
        logger.info("###")

    def get_DNN(self, num):
        checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
        model = nn.DataParallel(M.FastSpeech()).to(self.device)
        model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                      checkpoint_path), map_location=self.device)['model'])
        model.eval()
        return model

    def synthesis(self, model, text, alpha=1.0):
        text = np.array(text)
        text = np.stack([text])
        src_pos = np.array([i + 1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        if torch.cuda.is_available():
            sequence = torch.from_numpy(text).cuda().long()
            src_pos = torch.from_numpy(src_pos).cuda().long()
        else:
            sequence = torch.from_numpy(text).long()
            src_pos = torch.from_numpy(src_pos).long()

        with torch.no_grad():
            _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
        return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

    def get_data(self, texts):
        data_list = list()
        for text in texts:
            data_list.append(text.text_to_sequence(text, hp.text_cleaners))
        return data_list


    def gpt_generate(self, jsondata):
        logger.info(jsondata)
        # test1 = "I am very happy to see you again!"
        # test2 = "Durian model is a very good speech synthesis!"
        # test3 = "When I was twenty, I fell in love with a girl."
        # test4 = "I remove attention module in decoder and use average pooling to implement predicting r frames at once"
        # test5 = "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted."
        # test6 = "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
        content = jsondata['text']

        textbin = text.text_to_sequence(content, hp.text_cleaners)
        _, mel_cuda = self.synthesis(self.model, textbin, args.alpha)
        temp_file = io.BytesIO()
        waveglow.inference.inference(
            mel_cuda, self.WaveGlow, temp_file)
        return temp_file

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        text = req.get_param('text', True)
        # content = req.get_param('2', True)
        # clean_title = shortenlines(title)
        # clean_content = cleanall(content)
        torch.cuda.set_device(int(device_affinity))
        resp.content_type = 'audio/*'
        resp.stream = self.gpt_generate({"text": text})
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
        torch.cuda.set_device(int(device_affinity))
        print('device:', self.device)
        resp.content_type = 'audio/*'
        resp.stream = self.gpt_generate(jsondata)

if __name__ == "__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')