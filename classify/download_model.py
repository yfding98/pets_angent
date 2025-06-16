import timm
import torch

# 下载并保存模型权重
model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True)
torch.save(model.state_dict(), 'mobilenetv3_small_100.lamb_in1k.pth')