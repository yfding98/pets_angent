import argparse

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from models import BGERerankerWrapper

parser = argparse.ArgumentParser(description="启动 BgeM3 服务")
parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
parser.add_argument("--model-path", type=str, default="BAAI/bge-reranker-v2-m3", help="模型路径")
args = parser.parse_args()
app = FastAPI()
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
class RerankResultItem(BaseModel):
    index: int
    relevance_score: float
class RerankResponse(BaseModel):
    results: List[RerankResultItem]
    model_name: str

@app.post("/v1/rerank")
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    scores = model.rerank(request.query, request.documents)

    results = [
        RerankResultItem(
            index=i,
            relevance_score=score
        ) for i, score in enumerate(scores)
    ]

    return RerankResponse(
        results=results,
        model_name=model.model_name
    )



@app.on_event("startup")
def load_model():
    global model
    model = BGERerankerWrapper(model_path=args.model_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)