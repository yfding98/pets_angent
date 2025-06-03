FROM registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py38

ENV PATH="/opt/conda/envs/py38/bin:$PATH"

RUN apt update && \
    apt clean

RUN pip install --upgrade pip

WORKDIR /app
COPY ./requirements.txt /app/
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
