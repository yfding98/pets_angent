import argparse
import json
import random

import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse,StreamingResponse
import uvicorn
import requests
from httpx import AsyncClient
from starlette.middleware.cors import CORSMiddleware

from common.custom_exception import CustomException
from app.schema import ChatCompletionRequest, ChatCompletionImageRequest, ServerConfig

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


def load_config(file_path="config.yaml"):
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(e)

config = load_config()

VLLM_CHAT_SERVER = ServerConfig(**config["server"]["VLLM_CHAT_SERVER"])
VLLM_IMAGE_SERVER = ServerConfig(**config["server"]["VLLM_IMAGE_SERVER"])
CALLBACK_SERVER = ServerConfig(**config["server"]["CALLBACK_SERVER"])

@app.post("/v1/pets/images-recognize")
async def images_recognize(request:ChatCompletionImageRequest,raw_request: Request):
    prompts_map = {
        "emotion": "你擅长分析宠物的情绪，根据识别项：[位置和角度,对称性,瞳孔大小,眼睑状态,眼神,嘴巴,嘴角,舌头,胡须,尾巴状态,身体姿势,毛发状态,尾巴毛发状态,爪子位置,所处的场景],给出每个识别项的检测结果，并根据这些识别项的结果，从[愤怒、悲伤、惊慌、平静、开心、满足、兴奋、好奇]中选取一个最高可能性的进行一个整体的情绪判断，然后给眼睛、嘴巴、胡须、耳朵的状态和给出的情绪的相关程度进行打分（百分制），最后给出情绪的判断理由，并提供和该宠物的互动建议。特别注意：所有这些都以json格式输出，主键分别是：detectionResults,emotion,correlationScore(内部的主键为mouth,eyes,ears,whiskers),reason,suggestion!",
        "breed": "你是专业的宠物品种分析专家，请根据图片识别出该宠物可能的品种及其概率（可能性由高到低给出三种），给出可能性最高的品种判断的理由，并给出可能性最高的品种的起源与发展、寿命与体格的简单介绍，请使用json格式输出，主键分别是：petCategoryName、origin、physique、analyse、recognizeDetailList,其中recognizeDetailList中的主键是petCategoryName和percentage！"
    }
    headers = {
        "Content-Type": "application/json"
    }
    client_data = await raw_request.json()
    # 提取 session_id 和 business_type
    session_id = client_data.get("recognizeId", "")
    scene = client_data.get("scene", "")

    # 构造转发给 vLLM 的 payload（去掉中间层字段）
    vllm_payload = {k: v for k, v in client_data.items() if k not in ["recognizeId", "scene"]}

    system_prompt = "你是petpal，来自杭州知几智能，是专业的ai宠物助手，"+ prompts_map[scene]

    if vllm_payload["messages"][0]["role"] != "system":
        # 往messages列表的首位插入系统角色消息
        vllm_payload["messages"].insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    elif vllm_payload["messages"][0]["role"] == "system":
        original_content = vllm_payload["messages"][0]["content"]
        if isinstance(original_content, list):
            original_content = original_content[0]["text"]
            new_content = original_content + "\n" + system_prompt
            vllm_payload["messages"][0]["content"] = [{"type": "text", "text": new_content}]
        elif isinstance(original_content, str):
            vllm_payload["messages"][0]["content"] = original_content + "\n" + system_prompt

    response = requests.post(VLLM_IMAGE_SERVER.url, json=vllm_payload, headers=headers)

    result = response.json()
    result = result["choices"][0]["message"]["content"]
    json_str = result.strip("```json").strip("```").strip()
    result = json.loads(json_str)

    # 给 correlation_score 分数添加噪声
    if "correlationScore" in result:
        for key,  value in result["correlationScore"].items():
            if 5 <= value <= 95:
                result["correlationScore"][key] += int(random.uniform(-5, 5))
    if "detectionResults" in result:
        detection_results = []
        for key, value in result["detectionResults"].items():
            detection_results.append({
                "key": key,
                "value": value
            })
        result["detectionResults"] = detection_results

    result_ ={
        "scene":scene,
        "recognizeId":session_id,
        "result":result
    }

    # 回调接口
    requests.post(CALLBACK_SERVER.url, json=result_)
    return JSONResponse(content={}, status_code=200)

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request:ChatCompletionRequest,raw_request: Request):
    """
    中间服务接口，代理到 vLLM 的 /v1/chat/completions。
    支持流式输出和 session_id 透传。
    """
    client_data = await raw_request.json()

    # 提取 session_id 和 business_type
    session_id = client_data.get("session_id", "")
    business_type = client_data.get("business_type", "")

    # 构造转发给 vLLM 的 payload（去掉中间层字段）
    vllm_payload = {k: v for k, v in client_data.items() if k not in ["session_id", "business_type"]}
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


    is_streaming = vllm_payload.get("stream", False)

    if is_streaming:
        async def upstream_stream():
            async with AsyncClient() as client:
                async with client.stream(
                        method=VLLM_CHAT_SERVER.method,
                        url=VLLM_CHAT_SERVER.url,
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
                                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                            except json.JSONDecodeError:
                                yield f"{line}\n\n"
                        else:
                            yield f"{line}\n\n"

        return StreamingResponse(
            upstream_stream(),
            media_type="text/event-stream",
            headers={'X-Accel-Buffering': 'no'}
        )

    else:
        async with AsyncClient() as client:
            response = await client.post(
                VLLM_CHAT_SERVER.url,
                json=vllm_payload,
                headers={"Content-Type": "application/json"},
                timeout=600
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            data = response.json()
            data["session_id"] = session_id
            return JSONResponse(content=data)


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
        print(f"Request: {request.method} {request.url} Exception: {e}")
        return JSONResponse(
            status_code=500,
            content= {
                "code": 500,
                "message": "系统异常"
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)