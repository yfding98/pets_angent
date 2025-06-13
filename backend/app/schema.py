from typing import Optional, List, Literal, Dict, Union

from openai.types.evals.create_eval_completions_run_data_source import SamplingParams
from pydantic import BaseModel, ConfigDict, Field
from openai.types.chat import ChatCompletionMessageParam

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    url: str = ""
    method:  str = "POST"


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="forbid")

class ResponseFormat(OpenAIBaseModel):
    # type must be "json_object" or "text"
    type: Literal["text", "json_object"]

class ChatCompletionRequest(OpenAIBaseModel):
    session_id:  Optional[str] = None
    business_type: Optional[str] = None
    stream: Optional[bool] = False
    model:  str
    messages: List[ChatCompletionMessageParam]

class ChatCompletionImageRequest(OpenAIBaseModel):
    recognizeId:  Optional[str] = None
    scene: Optional[str] = None
    model:  str
    messages: List[ChatCompletionMessageParam]