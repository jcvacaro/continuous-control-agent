ARG from_nvidia_gl=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04
ARG from_pytorch=pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

# stage 1: use nvidia gl just to copy resources in the main stage
FROM ${from_nvidia_gl} as glvnd

# stage 2: main
FROM ${from_pytorch}

# enable all nvidia capabilities
ENV NVIDIA_DRIVER_CAPABILITIES all

# copy nvidia opengl resources from previous stage
COPY --from=glvnd /usr/local/lib/x86_64-linux-gnu /usr/local/lib/x86_64-linux-gnu
COPY --from=glvnd /usr/local/lib/i386-linux-gnu /usr/local/lib/i386-linux-gnu
COPY --from=glvnd /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=glvnd /etc/ld.so.conf.d/glvnd.conf /etc/ld.so.conf.d/glvnd.conf
RUN ldconfig
ENV LD_LIBRARY_PATH /usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# required packages
RUN apt-get update && apt-get install -y \
        autoconf \
        automake \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        libffi-dev \
        libssl-dev \
        libtool \
        make \
        software-properties-common \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN cd / \
    && git clone https://github.com/udacity/deep-reinforcement-learning.git \
    && cd deep-reinforcement-learning/python \
    && pip install -e .
