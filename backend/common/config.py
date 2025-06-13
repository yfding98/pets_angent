from pathlib import Path
from typing import Optional
import logging
import logging.handlers
import os
import yaml
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    url: str = ""
    method:  str = "POST"
    model_name: Optional[str] = ""
    api_key: Optional[str] = "*"



def setup_logger(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger_config = config["logger"]


    # 创建日志目录（如果不存在）
    log_file = logger_config["log_file"]
    if os.name == 'nt':
        log_file_name = os.path.basename(log_file)
        log_file = os.path.join(os.getcwd(), "log", log_file_name)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    level = getattr(logging, logger_config["level"], logging.INFO)
    formatter = logging.Formatter(logger_config["format"])

    # 设置 RotatingFileHandler
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when=logger_config.get("when", "midnight"),
        backupCount=logger_config.get("backup_count", 7),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # 设置控制台日志（可选）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 配置 root logger
    logging.basicConfig(
        level=level,
        handlers=[file_handler, console_handler]
    )

    logging.info("Logger initialized.")

def load_config(file_path="config.yaml"):
    try:
        with open(file_path, "r",  encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(e)



configs = load_config()
setup_logger()

VLLM_CHAT_SERVER = ServerConfig(**configs["server"]["VLLM_CHAT_SERVER"])
VLLM_IMAGE_SERVER = ServerConfig(**configs["server"]["VLLM_IMAGE_SERVER"])
CALLBACK_SERVER = ServerConfig(**configs["server"]["CALLBACK_SERVER"])