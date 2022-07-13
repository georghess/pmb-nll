FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common

RUN add-apt-repository ppa:savoury1/ffmpeg4
RUN apt-get install -y --no-install-recommends pkg-config
RUN apt install -y --no-install-recommends ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev libmagickwand-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	 python3 get-pip.py && \
	 rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install cython
RUN pip install torchvision
RUN pip install setuptools>=46.1.3 numpy>=1.18.2 scipy>=1.4.1 opencv-python>=4.2.0.32 \
pyyaml>=5.3.1 cython>=0.29.16 pascal-voc-tools>=0.1.29 tqdm>=4.42.0\
prettytable>=0.7.2 ujson>=2.0.3 scikit-learn>=0.23.1 scikit-image>=0.17.2\
uncertainty-calibration>=0.0.7 seaborn>=0.10.1 pandas>=1.0.4 wand>=0.6.2\
numba pillow matplotlib wheel

RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo

RUN pip install -e detectron2_repo
