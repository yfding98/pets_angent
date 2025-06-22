from typing import List

from FlagEmbedding import FlagReranker
import os

class BGERerankerWrapper:
    def __init__(self, model_path: str = "BAAI/bge-reranker-v2-m3"):
        self.device = os.getenv("DEVICE", "musa")  # 支持 CUDA/MUSA 设备
        self.use_fp16 = os.getenv("USE_FP16", "true").lower() == "true"  # FP16 推理
        self.model_name = os.path.basename(model_path)

        # 初始化 reranker 模型
        self.model = FlagReranker(
            model_name_or_path=model_path,
            use_fp16=self.use_fp16,
            device=self.device
        )

    def rerank(self, query: str, docs: List[str]):
        pairs = [[query, doc] for doc in docs]
        scores = self.model.compute_score(pairs)
        return [float(score) for score in scores]  # 确保返回 float 列表