import asyncio
import importlib
import inspect
import json
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Set, Optional

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm
import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
logger = init_logger(__name__)

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)

class ChatCompletionRequestSession(ChatCompletionRequest):
    session_id: Optional[str] = None
    business_type: Optional[str] = None


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequestSession, raw_request: Request):
    session_id = request.session_id

    generator = await openai_serving_chat.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content={
                "session_id": session_id,
                "error": generator.model_dump()
            },
            status_code=generator.code
        )

    if request.stream:
        async def stream_wrapper():
            try:
                async for item in generator:
                    if isinstance(item, str):
                        # 处理已序列化的字符串
                        logger.debug(f"Received string: {item}")
                        try:
                            data = json.loads(item[6:])  # 去掉 "data: "
                        except Exception:
                            yield item  # 原样返回
                            continue
                        data["session_id"] = session_id
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    else:
                        # 处理 Pydantic 模型
                        item_dict = item.model_dump()
                        item_dict["session_id"] = session_id
                        yield f"data: {json.dumps(item_dict, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                # 客户端中断连接，属于正常行为，仅记录 info 日志
                logger.info("Client disconnected during streaming.")
                # 必须重新 raise，否则 uvicorn 会卡住
                raise
            except Exception as e:
                logger.error(f"Unexpected error during streaming: {e}", exc_info=True)
                # 可选：发送错误信息给前端
                yield f"data: {json.dumps({'error': 'Internal server error'}, ensure_ascii=False)}\n\n"
                raise

        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    else:
        assert isinstance(generator, ChatCompletionResponse)
        response_data = generator.model_dump()
        response_data["session_id"] = session_id
        return JSONResponse(content=response_data)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info("vLLM API server version %s", vllm.__version__)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    openai_serving_chat = OpenAIServingChat(engine, served_model_names,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names, args.lora_modules)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
