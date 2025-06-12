FROM python:3.11-slim

ENV PATH="/opt/conda/envs/py38/bin:$PATH"

RUN apt update && \
    apt clean

RUN pip install --upgrade pip

WORKDIR /workspace
COPY ./requirements.txt /workspace/
COPY ./app /workspace/app
COPY ./common  /workspace/common
COPY ./config.yaml /workspace/
COPY ./main.py /workspace/
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
