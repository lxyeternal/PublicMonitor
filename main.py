import os
import sys
import logging
from services.event_extractor import EventExtractor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主函数"""

    input_file = "test/test.txt"
    input_id = "123456"
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info("舆情事件提取系统启动")
    logger.info(f"输入文件: {input_file}")
    
    # 初始化事件提取器
    extractor = EventExtractor()
    
    # 从文件中提取事件
    result = extractor.extract_events_from_file(input_file, input_id)
    
    if result:
        logger.info(f"成功提取 {len(result.get('events', []))} 个事件")
    else:
        logger.error("事件提取失败")
    
    logger.info("舆情事件提取系统结束")

if __name__ == "__main__":
    main() 