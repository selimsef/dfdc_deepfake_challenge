ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Setting noninteractive build, setting up tzdata and configuring timezones
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 nano mc glances vim git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install cython
RUN conda install cython -y && conda clean --all

# Installing APEX
RUN pip install -U pip
RUN git clone https://github.com/NVIDIA/apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./apex
RUN apt-get update -y
RUN apt-get install build-essential cmake -y
RUN apt-get install libopenblas-dev liblapack-dev -y
RUN apt-get install libx11-dev libgtk-3-dev -y
RUN pip install dlib
RUN pip install facenet-pytorch
RUN pip install albumentations==1.0.0 timm==0.4.12 pytorch_toolbelt tensorboardx
RUN pip install cython jupyter  jupyterlab ipykernel matplotlib tqdm pandas

# download pretraned Imagenet models
RUN apt install wget
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth -P /root/.cache/torch/hub/checkpoints/
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth -P /root/.cache/torch/hub/checkpoints/

# Setting the working directory
WORKDIR /workspace

# Copying the required codebase
COPY . /workspace

RUN chmod 777 preprocess_data.sh
RUN chmod 777 train.sh
RUN chmod 777 predict_submission.sh

ENV PYTHONPATH=.

CMD ["/bin/bash"]

