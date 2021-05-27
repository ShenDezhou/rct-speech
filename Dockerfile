FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
USER root
WORKDIR /workspace
COPY . /workspace
RUN pip install -r requirements.txt
RUN apt update && apt install libsndfile1 -y
CMD ["python3"]
