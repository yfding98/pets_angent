from FlagEmbedding import BGEM3FlagModel  # 替换为 FlagEmbedding 的导入方式
import os


class BGEM3FlagWrapper:
    def __init__(self, model_path: str = "BAAI/bge-m3"):
        # 配置模型参数
        self.device = os.getenv("DEVICE", "musa")  # 支持通过环境变量指定设备
        self.use_fp16 = os.getenv("USE_FP16", "true").lower() == "true"  # 是否启用 FP16

        # 初始化 BGEM3FlagModel
        self.model = BGEM3FlagModel(
            model_name_or_path=model_path,
            use_fp16=self.use_fp16,
            device=self.device
        )

    def encode(self, texts, batch_size=512, max_length=8192):
        # 调用 FlagEmbedding 的 encode 方法
        embeddings = self.model.encode(texts,batch_size, max_length)
        return embeddings  # FlagEmbedding 默认返回 numpy 数组，无需额外转换