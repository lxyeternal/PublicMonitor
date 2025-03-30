import os
import json
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """文本处理服务，负责读取文本文件并进行基础处理"""
    
    def __init__(self):
        """初始化文本处理器"""
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "text_processor.log"), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def read_text_file(self, file_path):
        """
        读取文本文件
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            文件内容
        """
        logger.info(f"开始读取文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"成功读取文件，内容长度: {len(content)} 字符")
            return content
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        对文本进行预处理
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        logger.info("开始文本预处理")
        
        # 基础预处理：去除多余空白字符
        processed_text = ' '.join(text.split())
        
        logger.info(f"文本预处理完成，处理后长度: {len(processed_text)} 字符")
        return processed_text
    
    def load_prompt(self, prompt_name):
        """
        加载提示词模板
        
        Args:
            prompt_name: 提示词文件名
            
        Returns:
            提示词模板内容
        """
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts", f"{prompt_name}.txt"
        )
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            return prompt
        except Exception as e:
            logger.error(f"加载提示词模板失败: {e}")
            return "" 