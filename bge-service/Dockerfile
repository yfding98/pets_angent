FROM registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py38

WORKDIR /app
COPY server/requirements.txt /app/
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple