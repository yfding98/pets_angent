import argparse
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from models import BGEM3FlagWrapper

# 解析命令行参数
parser = argparse.ArgumentParser(description="启动 BgeM3 服务")
parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
parser.add_argument("--model-path", type=str, default="BAAI/bge-m3", help="模型路径")
args = parser.parse_args()

app = FastAPI()
model = BGEM3FlagWrapper(args.model_path)

class EmbeddingRequest(BaseModel):
    input: List[str]

@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    embeddings = model.encode(request.input,batch_size=512, max_length=8192)['dense_vecs']
    return {
        "data": [{"embedding": emb.tolist()} for emb in embeddings],
        "model": "bge-m3",
        "usage": {"total_tokens": len(request.input)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)