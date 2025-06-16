FROM python:3.11-slim

ENV PATH="/opt/conda/envs/py38/bin:$PATH"

RUN apt update && \
    apt clean

RUN pip install --upgrade pip

WORKDIR /workspace
COPY . /workspace
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["python", "prepare_model.py"]
