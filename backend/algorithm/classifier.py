# classifier.py
import logging
import os
import sys
import time

import timm
import torch
import torch_musa
import json
from PIL import Image
from torchvision import transforms
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.config import BASE_DIR

MODEL_NAME = 'mobilenetv3_small_100.lamb_in1k'
LOCAL_WEIGHT_PATH = os.path.join(BASE_DIR, "models", "mobilenetv3_classification_image-1k.pth")
class AnimalClassifier:
    def __init__(self, model_path, class_index_path):
        """
        初始化分类器。
        - 加载TorchScript模型。
        - 加载类别索引。
        - 定义图像预处理转换。
        """
        print("正在初始化分类器...")
        # 1. 加载优化的TorchScript模型
        self.device = torch.device("musa")
        # self.model = torch.jit.load(model_path, map_location=self.device)
        # self.model = timm.create_model(MODEL_NAME, pretrained=True)

        self.model = timm.create_model(MODEL_NAME, pretrained=False)
        # 3. 加载本地权重
        state_dict = torch.load(LOCAL_WEIGHT_PATH, map_location=self.device)

        # 4. 加载权重到模型中（严格匹配或非严格匹配）
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print("Warning: Some weights not matched exactly. Trying non-strict loading...")
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()  # 确保是评估模式
        print(f"模型已加载到设备: {self.device}")

        # 2. 加载并解析类别索引
        with open(class_index_path) as f:
            class_idx = json.load(f)
        self.classes = [class_idx[str(k)][1] for k in range(len(class_idx))]

        # 3. 定义固定的图像转换 (基于prepare_model.py的输出)
        # MobileNetV3的典型值:
        self.transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # 4. 定义动物类别范围
        self.animal_categories = {
            "家犬 (Domestic Dog)": list(range(151, 269)),
            "猫科动物 (Cat)": list(range(281, 294)),
            "宠物鸟 (Pet Bird)": [88, 89, 90],
            "小型宠物 (Small Pet)": [333, 337],
            # ... 你可以添加其他广义类别
        }
        # 创建一个包含所有动物索引的集合，用于快速判断
        self.all_animal_indices = set()
        for indices in self.animal_categories.values():
            self.all_animal_indices.update(indices)

    def _classify_animal_from_index(self, pred_index):
        if pred_index in self.animal_categories["家犬 (Domestic Dog)"]:
            return "家犬 (Domestic Dog)"
        if pred_index in self.animal_categories["猫科动物 (Cat)"]:
            return "家猫 (Domestic Cat)" if 281 <= pred_index <= 285 else "野生猫科 (Wild Cat)"
        if pred_index in self.animal_categories["宠物鸟 (Pet Bird)"]:
            return "宠物鸟 (Pet Bird)"
        if pred_index in self.animal_categories["小型宠物 (Small Pet)"]:
            return "小型宠物 (Small Pet)"
        return "其他动物 (Other Animal)"

    def predict(self, image_bytes, top_k=5):
        """
        对输入的图片字节流进行预测。
        :param image_bytes: 图片文件的二进制内容。
        :param top_k: 返回前k个最可能的类别。
        :return: 一个包含预测结果的字典。
        """
        # 从字节流加载图片
        start_time= time.time()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"无法打开图片: {e}"}

        # 图像预处理
        input_tensor = self.transforms(img).unsqueeze(0).to(self.device)

        # 执行推理 (使用no_grad以提高性能并减少内存占用)
        with torch.no_grad():
            output = self.model(input_tensor)

        # 应用softmax获取概率
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 获取top-k结果
        top_k_prob, top_k_indices = torch.topk(probabilities, top_k)

        # 格式化输出
        results = []
        is_animal_detected = False
        for i in range(top_k_prob.size(0)):
            prob = top_k_prob[i].item() * 100
            idx = top_k_indices[i].item()
            label = self.classes[idx].replace('_', ' ')

            result_item = {
                "label": label,
                "probability": f"{prob:.2f}%",
                "isAnimal": False,
                "animalCategory": None
            }

            if idx in self.all_animal_indices:
                is_animal_detected = True
                result_item["isAnimal"] = True
                result_item["animalCategory"] = self._classify_animal_from_index(idx)

            results.append(result_item)
        cost_time = time.time() - start_time
        logging.info(f"[category]图片预分类完成，耗时：{cost_time:.2f}秒")
        return {
            "isAnimal": is_animal_detected,
            "predictions": results
        }