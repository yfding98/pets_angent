# prepare_model.py
import logging

import torch
import timm
import os
from urllib.request import urlopen

from common.config import BASE_DIR

# --- 配置 ---
MODEL_NAME = 'mobilenetv3_small_100.lamb_in1k'
MODEL_DIR = 'models'
BASE_MODEL_PATH = os.path.join(BASE_DIR,MODEL_DIR, 'mobilenetv3_classification_image-1k.pth')
SCRIPTED_MODEL_PATH = os.path.join(BASE_DIR,MODEL_DIR, 'mobilenetv3_scripted.pt')
CLASS_INDEX_PATH = os.path.join(BASE_DIR,MODEL_DIR, 'imagenet_class_index.json')


def prepare_environment():
    """
    执行所有一次性准备工作：下载模型，导出为TorchScript，下载类别索引。
    """
    print("开始准备模型和环境...")

    # # 2. 加载timm模型
    # logging.info(f"从timm加载预训练模型: {MODEL_NAME}...")
    # model = timm.create_model(MODEL_NAME, pretrained=True)
    # torch.save(model.state_dict(), os.path.join(BASE_DIR,MODEL_DIR,'mobilenetv3_classification_image-1k.pth'))
    # model.eval()  # 设置为评估模式
    #
    # # 3. 获取并打印数据预处理配置 (这些值将在服务器代码中硬编码)
    # data_config = timm.data.resolve_model_data_config(model)
    # logging.info("\n--- 模型数据配置 (请将这些值用于推理服务) ---")
    # print(f"Input size: {data_config['input_size']}")
    # print(f"Interpolation: {data_config['interpolation']}")
    # print(f"Mean: {data_config['mean']}")
    # print(f"Std: {data_config['std']}")
    # print("---------------------------------------------------\n")
    #
    # # 4. 创建一个示例输入，用于追踪模型
    # # (N, C, H, W) -> (1, 3, 224, 224) for mobilenetv3
    # example_input = torch.randn(1, *data_config['input_size'])
    #
    # os.makedirs(os.path.dirname(SCRIPTED_MODEL_PATH), exist_ok=True)
    #
    # # 5. 使用TorchScript JIT追踪模型
    # logging.info("正在将模型转换为TorchScript...")
    # try:
    #     scripted_model = torch.jit.trace(model, example_input)
    #     scripted_model.save(SCRIPTED_MODEL_PATH)
    #     logging.info(f"TorchScript模型已成功保存到: {SCRIPTED_MODEL_PATH}")
    # except Exception as e:
    #     logging.error(f"模型导出失败: {e}")
    #     return

    # 6. 下载并保存ImageNet类别索引文件
    if not os.path.exists(CLASS_INDEX_PATH):
        logging.info("正在下载ImageNet类别索引...")
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        with urlopen(url) as response:
            class_idx_data = response.read()
        with open(CLASS_INDEX_PATH, 'wb') as f:
            f.write(class_idx_data)
        logging.info(f"类别索引已保存到: {CLASS_INDEX_PATH}")
    else:
        print("类别索引文件已存在。")

    print("\n所有准备工作完成！")


if __name__ == '__main__':
    prepare_environment()