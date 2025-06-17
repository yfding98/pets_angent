FROM registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py310

ENV PATH="/opt/conda/envs/py310/bin:$PATH"

RUN apt update && \
    apt clean

RUN pip install --upgrade pip

WORKDIR /workspace
COPY . /workspace
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python prepare_model.py
