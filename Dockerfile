FROM mindie:dev-2.0.RC1-B020-800I-A2-py311-ubuntu22.04-aarch64 AS base

RUN sh -c "echo '127.0.0.1 $(hostname)' >> /etc/hosts";\
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy main restricted" > /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy multiverse" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy universe" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy-security main restricted" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy-security multiverse" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy-security universe" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy-updates main restricted" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy-updates multiverse" >> /etc/apt/sources.list && \
    echo "deb [trusted=yes] http://mirrors.tools.huawei.com/ubuntu-ports/ jammy-updates universe" >> /etc/apt/sources.list

RUN set -eux; \
    apt update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        netbase \
        curl \
        wget \
        tzdata \
        wget \
        vim \
        autoconf \
        automake \
        libtool \
        git \
        tcl \
        patch \
        libnuma-dev \
        flex \
        tclsh \
        git-lfs \
        pkg-config \
        aria2; \
    apt-get clean;

RUN pip config set global.index-url 'https://mirrors.tools.huawei.com/pypi/simple/'; \
    pip config set global.trusted-host "mirrors.tools.huawei.com pypi.org files.pythonhosted.org ms-release.obs.cn-north-4.myhuaweicloud.com"

FROM base AS ms_vllm

WORKDIR /workspace/mindspore-vllm
COPY . /workspace/mindspore-vllm

ENV VLLM_TARGET_DEVICE=empty
RUN cd vllm-v0.6.6.post1; \
    pip install -e .; \
    cd ..

RUN pip install -r requirements.txt --progress-bar on && \
    pip install -e . && \
    pip uninstall torch torch-npu torchvision -y
