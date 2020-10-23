FROM nvidia/cudagl:10.1-devel

## cudnn
ENV CUDNN_VERSION 7.6.5.32
RUN apt-get update && \
    apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda10.1 libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 && \
    apt-mark hold libcudnn7

## sshd
RUN apt-get install -y --no-install-recommends openssh-server && mkdir /var/run/sshd && \
    echo 'root:password' | chpasswd && \
    sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo "export VISIBLE=now" >> /etc/profile
ENV NOTVISIBLE="in users profile"
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

## python
RUN apt-get install -y --no-install-recommends python3-dev python3-pip git vim wget && \
    python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install tensorflow numpy==1.18.4 opencv-python gdown wheel torch==1.4.0 torchvision==0.5.0

## vibe
RUN apt-get install -y --no-install-recommends unzip ffmpeg libsm6 libxext-dev libxrender-dev freeglut3-dev && \
    python3 -m pip install yacs scipy numba smplx==0.1.13 PyYAML joblib trimesh pillow pyrender progress filterpy matplotlib \
    scikit-image scikit-video llvmlite git+https://github.com/mattloper/chumpy.git \
    git+https://github.com/mkocabas/yolov3-pytorch.git git+https://github.com/mkocabas/multi-person-tracker.git


## openpose 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libprotobuf-dev protobuf-compiler libopencv-dev libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev \
    libhdf5-dev libatlas-base-dev

RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz

ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

WORKDIR /openpose/build
RUN cmake -DBUILD_PYTHON=ON .. && make -j `nproc`
WORKDIR /openpose
