import argparse

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from models import BGERerankerWrapper

parser = argparse.ArgumentParser(description="启动 BgeM3 服务")
parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
parser.add_argument("--model-path", type=str, default="BAAI/bge-m3", help="模型路径")
args = parser.parse_args()
app = FastAPI()
model = BGERerankerWrapper(model_path=args.model_path,)

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

@app.post("/v1/rerank")
async def rerank_documents(request: RerankRequest):
    scores = model.rerank(request.query, request.documents)
    return {
        "scores": scores,
        "model": "bge-reranker-v2-m3",
        "usage": {"total_pairs": len(request.documents)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)