FROM registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py38

WORKDIR /app
COPY ./requirements.txt /app/
# 安装依赖
# RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi==0.115.12 \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple pydantic==2.10.5  \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uvicorn==0.33.0  \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple FlagEmbedding==1.3.5