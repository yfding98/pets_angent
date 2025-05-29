from sentence_transformers import SentenceTransformer
import torch


class BgeM3Model:
    def __init__(self, model_path: str = "BAAI/bge-m3"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path).to(self.device)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).cpu().numpy()