from typing import List

from FlagEmbedding import FlagReranker
import os


class BGERerankerWrapper:
    def __init__(self, model_path: str = "BAAI/bge-reranker-v2-m3"):
        self.device = os.getenv("DEVICE", "musa")  # 支持 CUDA/MUSA 设备
        self.use_fp16 = os.getenv("USE_FP16", "true").lower() == "true"  # FP16 推理

        # 初始化 reranker 模型
        self.model = FlagReranker(
            model_name_or_path=model_path,
            use_fp16=self.use_fp16,
            device=self.device
        )

    def rerank(self, query: str, docs: List[str]):
        # 调用 FlagEmbedding 的 rerank 方法
        scores = self.model.compute_score([[query, doc] for doc in docs])
        return scores  # 返回相似度分数列表