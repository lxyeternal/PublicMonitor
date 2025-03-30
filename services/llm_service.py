import logging
import time
import json
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LLMService:
    """Azure OpenAI服务实现"""
    
    def __init__(self):
        """初始化LLM服务"""
        # 加载配置
        self.config = self._load_config()
        self.max_attempts = 5
        self.init_client()
        
    def _load_config(self) -> Dict[str, Any]:
        """从配置文件加载设置"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "llmsettings.json"
        )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"成功从 {config_path} 加载LLM配置")
                return config
        except Exception as e:
            logger.error(f"无法加载LLM配置: {e}")
            return {}
        
    def init_client(self):
        """初始化Azure OpenAI客户端"""
        try:
            from openai import AzureOpenAI
            
            self.client = AzureOpenAI(
                api_key=self.config.get("azure_api_key"),
                api_version=self.config.get("azure_api_version"),
                azure_endpoint=self.config.get("azure_api_base")
            )
            logger.info("成功初始化Azure OpenAI客户端")
            
        except Exception as e:
            logger.error(f"初始化Azure OpenAI客户端失败: {e}")
            self.client = None
    
    def query(self, 
                   messages: List[Dict[str, str]], 
                   model: str = None,
                   temperature: float = 0,
                   max_tokens: int = 10000,
                   response_format: Dict = None,
                   **kwargs) -> str:
        """
        查询Azure OpenAI
        
        Args:
            messages: 输入消息
            model: 模型名称，如果为None则使用配置中的deployment_name
            temperature: 温度参数
            max_tokens: 最大生成token数
            response_format: 响应格式
            
        Returns:
            LLM响应文本
        """
        if not self.client:
            logger.error("Azure OpenAI客户端未初始化")
            return ""
            
        # 使用指定模型或配置中的部署名称
        deployment_name = model if model else self.config.get("deployment_name", "gpt-4o")
        
        # 记录当前时间戳，用于日志文件名
        timestamp = int(time.time())
        
        for attempt in range(self.max_attempts):
            try:
                # 准备请求参数
                params = {
                    "model": deployment_name,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"} if response_format is None else response_format,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    "seed": 42
                }
                
                # 合并额外参数
                params.update(kwargs)
                
                # 将输入保存到日志
                log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "llm_queries")
                os.makedirs(log_dir, exist_ok=True)
                
                log_file = os.path.join(log_dir, f"llm_query_{timestamp}_{attempt}.json")
                
                # 记录输入
                log_data = {
                    "timestamp": timestamp,
                    "attempt": attempt,
                    "input": {
                        "messages": messages,
                        "params": params
                    }
                }
                
                # 执行查询
                response = self.client.chat.completions.create(**params)
                response_text = response.choices[0].message.content
                
                # 记录输出
                log_data["output"] = {
                    "response": response_text
                }
                
                # 保存日志
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"LLM查询日志已保存至: {log_file}")
                
                return response_text
                
            except Exception as e:
                logger.warning(f"Azure OpenAI查询失败（尝试 {attempt+1}/{self.max_attempts}）: {e}")
                
                # 记录错误
                if 'log_data' in locals() and 'log_file' in locals():
                    log_data["error"] = str(e)
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, ensure_ascii=False, indent=2)
                        
                if attempt < self.max_attempts - 1:
                    # 指数退避
                    wait_time = (2 ** attempt) + 1
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大尝试次数，查询失败")
                    return ""
    
    async def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """获取文本嵌入"""
        if not self.client:
            logger.error("Azure OpenAI客户端未初始化")
            return []
            
        embedding_model = model if model else "text-embedding-ada-002"
        
        try:
            response = await self.client.embeddings.create(
                model=embedding_model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"获取嵌入失败: {e}")
            return [] 