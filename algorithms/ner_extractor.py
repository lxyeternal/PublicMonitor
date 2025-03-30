import logging
import spacy
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class NERExtractor:
    """命名实体识别器，负责从文本中提取命名实体"""
    
    def __init__(self):
        """初始化NER提取器"""
        self.setup_logging()
        self.load_model()
        
    def setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "ner_extractor.log"), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_model(self):
        """加载NER模型"""
        logger.info("加载SpaCy NER模型")
        try:
            # 加载英文模型
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy NER模型加载成功")
        except Exception as e:
            logger.error(f"加载SpaCy模型失败: {e}")
            logger.info("尝试下载SpaCy模型...")
            try:
                # 如果模型不存在，尝试下载
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy NER模型下载并加载成功")
            except Exception as e:
                logger.error(f"下载SpaCy模型失败: {e}")
                # 创建一个空的管道作为后备
                self.nlp = spacy.blank("en")
                logger.warning("使用空白模型作为后备")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表，每个实体包含ID、文本、类型和位置信息
        """
        logger.info("开始提取命名实体")
        
        # 使用SpaCy处理文本
        doc = self.nlp(text)
        
        # 提取实体
        entities = []
        entity_id_counter = 1
        
        # 实体类型映射
        entity_type_map = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "CARDINAL": "OTHER",
            "MONEY": "OTHER",
            "PERCENT": "OTHER",
            "PRODUCT": "OTHER",
            "EVENT": "OTHER",
            "WORK_OF_ART": "OTHER",
            "LAW": "OTHER",
            "LANGUAGE": "OTHER",
            "FAC": "LOCATION",
            "NORP": "OTHER"
        }
        
        # 处理实体
        for ent in doc.ents:
            entity_type = entity_type_map.get(ent.label_, "OTHER")
            
            # 检查是否已存在相同文本的实体
            existing_entity = next((e for e in entities if e["text"].lower() == ent.text.lower()), None)
            
            if existing_entity:
                # 如果实体已存在，添加新的提及
                existing_entity["mentions"].append({
                    "text": ent.text,
                    "position": [ent.start_char, ent.end_char]
                })
            else:
                # 如果实体不存在，创建新实体
                entity = {
                    "entity_id": f"E{entity_id_counter}",
                    "text": ent.text,
                    "type": entity_type,
                    "mentions": [{
                        "text": ent.text,
                        "position": [ent.start_char, ent.end_char]
                    }]
                }
                entities.append(entity)
                entity_id_counter += 1
        
        logger.info(f"提取到 {len(entities)} 个命名实体")
        return entities 