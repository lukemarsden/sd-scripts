#FROM nvidia/cuda:11.6.2-base-ubuntu20.04
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

RUN mkdir /app
WORKDIR /app

RUN apt-get update -y && apt-get install -y python3 python3-pip git unzip wget && \
    git clone https://github.com/bmaltais/kohya_ss && \
    cd kohya_ss && \
    pip install -r requirements.txt && \
    pip install bitsandbytes==0.41.1 && \
    mkdir sdxl && (cd sdxl; wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors)
