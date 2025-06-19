import argparse
import asyncio
import json
import logging
import os.path
import random
import time
import traceback
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,StreamingResponse
import uvicorn
from httpx import AsyncClient
from starlette.authentication import AuthCredentials, UnauthenticatedUser
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.middleware.cors import CORSMiddleware

from algorithm import classifier
from algorithm.classifier import AnimalClassifier
from common.config import BASE_DIR
from common.config import CALLBACK_SERVER, VLLM_IMAGE_SERVER, VLLM_CHAT_SERVER
from common.custom_exception import CustomException
from app.schema import ChatCompletionRequest, ChatCompletionImageRequest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)
parser = argparse.ArgumentParser(description="启动 模型接口处理 服务")
parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
args = parser.parse_args()
# 创建分类器单例
try:
    classifier = AnimalClassifier(model_path=os.path.join(BASE_DIR, 'models', 'mobilenetv3_scripted.pt'),
                                  class_index_path=os.path.join(BASE_DIR, 'models','imagenet_class_index.json'))
except Exception as e:
    raise RuntimeError(f"初始化分类器失败: {e}")



async def send_failure_callback(scene: str, recognize_id: str, error_message: str):
    """
    统一发送失败回调
    :param scene: 请求场景
    :param recognize_id: 唯一标识
    :param error_message: 错误信息
    """
    fail_result = {
        "scene": scene,
        "recognizeId": recognize_id,
        "status": "fail",
        "failMessage": f"服务异常,错误信息：{error_message}"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(CALLBACK_SERVER.host+CALLBACK_SERVER.url, json=fail_result)
            if response.status_code != 200:
                logging.error(f"Callback failed with status {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"Failed to send callback even after main error: {e}, traceback: {traceback.format_exc()}")

async def process_and_callback(client_data):
    try:
        prompts_map = {
            "emotion": f"你擅长分析宠物的情绪，根据识别项：[位置和角度,对称性,瞳孔大小,眼睑状态,眼神,嘴巴,嘴角,舌头,胡须,尾巴状态,身体姿势,毛发状态,尾巴毛发状态,爪子位置,所处的场景],给出每个识别项的检测结果，并根据这些识别项的结果，从[愤怒、悲伤、惊慌、平静、开心、满足、兴奋、好奇]中选取一个最高可能性的进行一个整体的情绪判断，然后给眼睛、嘴巴、胡须、耳朵的状态和给出的情绪的相关程度进行打分（百分制），最后给出情绪的判断理由，并提供和该宠物的互动建议。特别注意：所有这些都以json格式输出，主键分别是：detectionResults(内部主键是识别项的名称，无须转换英文),emotion,correlationScore(内部的主键为mouth,eyes,ears,whiskers),reason,suggestion!",
            "category": f"你是专业的宠物品种分析专家，请根据图片识别出该宠物可能的品种及其概率（可能性由高到低给出三种），给出可能性最高的品种判断的理由，并给出可能性最高的品种的起源与发展、寿命与体格的简单介绍，请使用json格式输出，主键分别是：petCategoryName、origin、physique、analyse、recognizeDetailList,其中recognizeDetailList中的主键是petCategoryName和percentage！注意petCategoryName的值要是中文（务必保证宠物类别的中英转化要准确）。"
        }
        scene = client_data.get("scene",  "")
        session_id = client_data.get("recognizeId", "")
        pet_info = client_data.get("petArchive", "")
        if scene not in prompts_map:
            raise CustomException(code=403, message="Invalid scene")
        if not session_id:
            raise CustomException(code=403, message="Invalid recognizeId")
        # 构造 payload、system prompt 等逻辑与原函数一致
        client_data["model"] = VLLM_IMAGE_SERVER.model_name
        vllm_payload = {k: v for k, v in client_data.items() if k not in ["recognizeId", "scene", "petArchive"]}
        system_prompt = "你是petpal，来自杭州知几智能，是专业的ai宠物助手，" + prompts_map[scene]

        # 先使用小模型对宠物图片进行分类
        image_url = vllm_payload["messages"][0]["content"][0]["image_url"]["url"]
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=10)
            if response.status_code != 200:
                raise CustomException(code=400, message="无法访问图片URL: {image_url}")

            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise CustomException(code=400, message=f"远程内容不是图片: {content_type}")

            image_bytes = response.content


        # 调用分类器
        results = classifier.predict(image_bytes)
        if "error" in results:
            raise CustomException(code=400, message=results["error"])
        elif results["isAnimal"]:
            # 遍历 results 的 predictions中 isAnimal 为 True 所有元素
            able_predictions = [prediction for prediction in results["predictions"] if prediction["isAnimal"]]
            category = able_predictions[0]["label"]
            # 生成品种的 提示词，是 able_predictions 的前三个的 品种和概率
            category_prompt = f"已知图片对应宠物的品种英文是({category})，"
            system_prompt = "你是petpal，来自杭州知几智能，是专业的ai宠物助手，"+ category_prompt + prompts_map[scene]
        else:
            raise CustomException(code=400, message="无法识别图片中的宠物")

        if vllm_payload["messages"][0]["role"] != "system":
            vllm_payload["messages"].insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        elif vllm_payload["messages"][0]["role"] == "system":
            original_content = vllm_payload["messages"][0]["content"]
            if isinstance(original_content, list):
                original_content = original_content[0]["text"]
                new_content = original_content + "\n" + system_prompt
                vllm_payload["messages"][0]["content"] = [{"type": "text", "text": new_content}]
            elif isinstance(original_content, str):
                vllm_payload["messages"][0]["content"] = original_content + "\n" + system_prompt
        start_time = time.time()
        async with httpx.AsyncClient(timeout=65.0) as client:
            try:
                response = await client.post(VLLM_IMAGE_SERVER.host+VLLM_IMAGE_SERVER.url, json=vllm_payload)
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                logging.error(f"Model service timeout, cost " + str(time.time() - start_time) + " s")
                raise CustomException(code=408, message="模型推理超时,请检查图片！")

        if response.status_code != 200:
            logging.error(f"Model service error")
            raise CustomException(code=response.status_code, message="模型推理失败！可能是图像异常，请检查图像质量")
        else:
            logging.info(f"Image2Txt model exec success! cost time: {time.time() - start_time} s")

        result = response.json()
        result = result["choices"][0]["message"]["content"]
        json_str = result.strip("```json").strip("```").strip()
        result = json.loads(json_str)

        # 给 correlation_score 分数添加噪声
        if "correlationScore" in result:
            for key, value in result["correlationScore"].items():
                if 5 <= value <= 95:
                    result["correlationScore"][key] += int(random.uniform(-5, 5))
        if "detectionResults" in result:
            detection_results = []
            for key, value in result["detectionResults"].items():
                detection_results.append({"key": key, "value": value})
            result["detectionResults"] = detection_results

        result_json = json.dumps(result, ensure_ascii=False)
        callback_data = {
            "scene": scene,
            "recognizeId": session_id,
            "result": result_json,
            "status": "success",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            r_ = await client.post(CALLBACK_SERVER.host+CALLBACK_SERVER.url, json=callback_data)
            if r_.status_code != 200:
                logging.error(f"Callback failed with status {r_.status_code}: {r_.text}")
            else:
                if r_.json()["code"] != "200":
                    logging.warning("Callback Server Inner Error"+r_.json()["message"])
                else:
                    logging.info(f"Callback successful: record_id = {session_id}")
    except CustomException as exc:
        logging.error(f"recognizeId:[{session_id}] CustomException: {exc.message}")
        raise HTTPException(status_code=exc.code, detail=exc.message)
    except httpx.ReadTimeout as exc:
        logging.error(f"recognizeId:[{session_id}] Exception:请求超时,{exc}")
        await send_failure_callback(scene, session_id, "网络异常，请求超时")
        raise HTTPException(status_code=504, detail="上游服务响应超时")
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
        error_msg = f"Model response format error: {exc}"
        logging.error(f"recognizeId:[{session_id}] Exception:{error_msg}")
        await send_failure_callback(scene, session_id, "识别失败")
        raise HTTPException(status_code=500, detail=error_msg)

    except Exception as exc:
        session_id = client_data.get("recognizeId", "")
        scene  = client_data.get("scene",  "")
        await send_failure_callback(scene, session_id, "未知异常")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
@app.post("/v1/pets/images-recognize",summary="图片识别接口")
async def images_recognize(request:ChatCompletionImageRequest,raw_request: Request):
    client_data = await raw_request.json()
    asyncio.create_task(process_and_callback(client_data))

    return JSONResponse(content={"status": "received"}, status_code=200)

@app.post("/v1/chat/completions", summary="大模型聊天接口")
async def proxy_chat_completions(request:ChatCompletionRequest,raw_request: Request):
    """
    中间服务接口，代理到 vLLM 的 /v1/chat/completions。
    支持流式输出和 session_id 透传。
    """
    client_data = await raw_request.json()

    # 提取 session_id 和 business_type
    session_id = client_data.get("session_id", "")
    business_type = client_data.get("business_type", "")
    pet_info = client_data.get("petArchive", "")
    client_data["model"] = VLLM_CHAT_SERVER.model_name

    # 构造转发给 vLLM 的 payload（去掉中间层字段）
    vllm_payload = {k: v for k, v in client_data.items() if k not in ["session_id", "business_type", "petArchive"]}
    base_info ="你是petpal,来自杭州知几科技"
    # 添加业务类型提示词逻辑（可选）
    prompt_map = {
        "Disease": "你是专业的宠物兽医顾问。请根据用户描述的症状，提供可能的疾病分析、轻重缓急判断，并建议是否需要尽快就医。同时解释常见病因和初步护理建议。",
        "Medication": "你是宠物药理专家。请根据用户提供的宠物症状或疾病，给出合理的药物选择建议（包括处方药与非处方药），说明剂量参考、使用方法、注意事项及可能的副作用。避免对未确诊情况做出直接用药指导。",
        "Dietary": "你是宠物营养师。请根据用户提供的宠物品种、年龄、体重及饮食内容，评估当前饮食结构是否合理，并提出优化建议。如涉及特殊需求（如减肥、过敏等），请提供个性化喂养方案。",
        "Beauty": "你是宠物美容护理专家。请根据用户提供的宠物品种、毛发类型和护理问题，提供日常清洁、刷毛频率、洗澡技巧、皮肤保养等建议。如涉及皮肤病或异常脱毛，请给予初步识别提示。",
        "FirstAid": "你是宠物急救专家。请根据用户描述的紧急情况（如中毒、骨折、烫伤等），提供第一时间应采取的措施，包括如何稳定宠物状态、何时必须送医、可否进行家庭处理等关键信息。",
        "Training": "你是宠物行为训练师。请根据用户的宠物品种、年龄和训练目标（如定点上厕所、听指令、社交训练等），提供科学的行为训练建议，包括正向激励方式、常见误区、训练周期预估等。",
        "Deworming": "你是宠物寄生虫防治专家。请根据宠物的生活环境（室内/户外）、年龄、驱虫历史等情况，推荐合适的驱虫药品、频率及使用方法，并解释体内/体外寄生虫的危害与预防策略。"
    }
    default_prompt = ",是一个ai宠物助手,请根据用户的问题,给出最合适的回答。"
    if vllm_payload["messages"][0]["role"] != "system":
        # 往messages列表的首位插入系统角色消息
        vllm_payload["messages"].insert(0, {"role": "system", "content": base_info +","+ prompt_map.get(business_type, default_prompt)})
    elif vllm_payload["messages"][0]["role"] == "system":
        original_content = vllm_payload["messages"][0]["content"]
        vllm_payload["messages"][0]["content"] = original_content + "\n" + prompt_map.get(business_type, default_prompt)

    logging.info(f"[RequestID: {session_id}] messages: {vllm_payload['messages']}")
    is_streaming = vllm_payload.get("stream", False)

    if is_streaming:
        full_response = []
        async def upstream_stream():
            async with AsyncClient() as client:
                async with client.stream(
                        method=VLLM_CHAT_SERVER.method,
                        url=VLLM_CHAT_SERVER.host+VLLM_CHAT_SERVER.url,
                        json=vllm_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=600
                ) as response:

                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail="Upstream error")

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        if line.startswith("data:"):
                            try:
                                data_str = line[5:].strip()
                                data = json.loads(data_str)
                                data["session_id"] = session_id
                                full_response.append(data["choices"][0]["delta"])
                                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                            except json.JSONDecodeError:
                                yield f"{line}\n\n"
                        else:
                            yield f"{line}\n\n"
                    logging.info(f"[RequestID: {session_id}] Full Response: {full_response}")


        return StreamingResponse(
            upstream_stream(),
            media_type="text/event-stream",
            headers={'X-Accel-Buffering': 'no'}
        )

    else:
        async with AsyncClient() as client:
            response = await client.post(
                VLLM_CHAT_SERVER.host+VLLM_CHAT_SERVER.url,
                json=vllm_payload,
                headers={"Content-Type": "application/json"},
                timeout=600
            )

            if response.status_code != 200:
                logging.error(f"Request: {request.method} {request.url} Exception: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)

            data = response.json()
            data["session_id"] = session_id
            return JSONResponse(content=data)


@app.post("/v1/pets/flash-recognize", summary="快速识别图片是否有宠物")
async def classify(
        image_url: Optional[str] = Body(None,embed=True, description="图片的URL地址"),
):
    """
    返回
    isAnimal:是否可能是宠物
    predictions: 检测结果列表，前五种可能性
        "label": 具体的品种名,
        "probability": 概率,
        "isAnimal": False,
        "animalCategory": 动物品种大类
    """
    # 校验是否提供了有效的输入
    if not image_url:
        raise CustomException(code=400,message="必须提供图片URL。")
    try:
        # 来自远程URL
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=10)
            if response.status_code != 200:
                raise CustomException(code=400,message="无法访问图片URL: {image_url}")

            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise CustomException(code=400,message=f"远程内容不是图片: {content_type}")

            image_bytes = response.content

        # 调用分类器
        results = classifier.predict(image_bytes)

        if "error" in results:
            raise HTTPException(status_code=422, detail=results["error"])

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时发生未知错误: {str(e)}")

class DummyAuthBackend:
    async def authenticate(self, request: Request):
        return AuthCredentials([]), UnauthenticatedUser()


app.add_middleware(AuthenticationMiddleware, backend=DummyAuthBackend())

# 异常处理中间件
@app.middleware("http")
async def exception_handler_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except CustomException as ce:
        print(f"Request: {request.method} {request.url} CustomException: {ce}")
        return JSONResponse(
            status_code=200,  # 业务异常返回200，保持接口调用语义
            content= {
                "code": ce.code,
                "message": ce.message
            }
        )
    except Exception as e:
        # 处理系统异常
        logging.error(f"Request: {request.method} {request.url} Exception: {e}")
        return JSONResponse(
            status_code=500,
            content= {
                "code": 500,
                "message": "系统异常"
            }
        )

logger = logging.getLogger("http")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # 转为毫秒
    logger.info(f"{request.client.host} {request.method} {request.url.path} {response.status_code} {process_time:.2f}ms")
    return response
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation Error: {exc} in request {request.url}")
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": "请求数据格式错误",
            "errors": exc.errors()  # 可选：显示详细错误信息
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)