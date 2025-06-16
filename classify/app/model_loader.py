import timm
import torch
from timm.data import resolve_model_data_config, create_transform

def load_local_model(model_name="mobilenetv3_small_100.lamb_in1k", model_path="../models/mobilenetv3_small_100.lamb_in1k.pth"):
    """
    从本地路径加载 timm 模型
    """
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 加载模型与 transform
model = load_local_model()
data_config = resolve_model_data_config(model)
transform = create_transform(**data_config, is_training=False)

# 加载 ImageNet 标签
import json
with open("../assets/imagenet_class_index.json") as f:
    class_idx = json.load(f)
classes = [class_idx[str(k)][1] for k in range(len(class_idx))]