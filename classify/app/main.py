# app.py (FastAPI version)

import os
import io
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from ..classifier import AnimalClassifier

# --- 全局配置与模型加载 ---
print("启动 FastAPI 服务器，正在加载模型...")
MODEL_PATH = os.path.join('models', 'mobilenetv3_scripted.pt')
CLASS_INDEX_PATH = os.path.join('models', 'imagenet_class_index.json')

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDEX_PATH):
    raise RuntimeError("模型文件未找到！请先运行 'python prepare_model.py'。")

# 创建分类器单例
try:
    classifier = AnimalClassifier(model_path=MODEL_PATH, class_index_path=CLASS_INDEX_PATH)
except Exception as e:
    raise RuntimeError(f"初始化分类器失败: {e}")

print("模型加载完毕，服务器准备就绪。")

# 创建 FastAPI 应用实例
app = FastAPI(
    title="动物分类器 API",
    description="上传一张图片，API将判断其中是否包含动物，并给出分类。",
    version="1.0.0"
)


@app.post("/predict", summary="上传图片文件或指定图片URL进行分类")
async def predict(
        # image: UploadFile = File(None, description="要进行分类的图片文件"),
        image_url: Optional[str] = Query(None, description="图片的URL地址"),
):
    """
    支持以下任意一种输入方式：

    - **image**: 图片文件 (e.g., .jpg, .png)
    - **image_url**: 远程图片链接

    返回 Top-5 的分类结果。
    """

    # 校验是否提供了有效的输入
    if not image_url:
        raise HTTPException(status_code=400, detail="必须提供图片URL。")
    try:
        # 来自远程URL
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"无法访问图片URL: {image_url}")

            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"远程内容不是图片: {content_type}")

            image_bytes = response.content

        # 调用分类器
        results = classifier.predict(image_bytes)

        if "error" in results:
            raise HTTPException(status_code=422, detail=results["error"])

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时发生未知错误: {str(e)}")

# 你可以添加一个启动事件，虽然在这里不是必须的，但是一个好习惯
@app.on_event("startup")
async def startup_event():
    print("FastAPI 应用已启动。")
    # 可以在这里执行一些启动任务，比如连接数据库等


@app.on_event("shutdown")
def shutdown_event():
    print("FastAPI 应用正在关闭。")