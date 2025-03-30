import logging
import os
import spacy
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SRLExtractor:
    """语义角色标注器，负责提取事件的基本要素"""
    
    def __init__(self):
        """初始化语义角色标注器"""
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
                logging.FileHandler(os.path.join(log_dir, "srl_extractor.log"), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_model(self):
        """加载依存句法分析模型"""
        logger.info("加载SpaCy依存句法分析模型")
        try:
            # 加载英文模型
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy依存句法分析模型加载成功")
        except Exception as e:
            logger.error(f"加载SpaCy模型失败: {e}")
            # 创建一个空的管道作为后备
            self.nlp = spacy.blank("en")
            logger.warning("使用空白模型作为后备")
    
    def extract_srl(self, text: str, triggers: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取文本中的语义角色
        
        Args:
            text: 输入文本
            triggers: 事件触发词列表
            entities: 实体列表
            
        Returns:
            事件基本要素列表
        """
        logger.info("开始提取语义角色")
        
        # 使用SpaCy处理文本
        doc = self.nlp(text)
        
        # 提取事件基本要素
        events = []
        event_id_counter = 1
        
        # 为每个触发词创建一个事件
        for trigger in triggers:
            # 找到触发词在文档中的位置
            trigger_start = trigger["position"][0]
            trigger_end = trigger["position"][1]
            
            # 找到包含触发词的句子
            trigger_sentence = None
            for sent in doc.sents:
                if sent.start_char <= trigger_start and sent.end_char >= trigger_end:
                    trigger_sentence = sent
                    break
            
            if not trigger_sentence:
                continue
            
            # 找到触发词对应的token
            trigger_token = None
            for token in trigger_sentence:
                if token.idx <= trigger_start and token.idx + len(token.text) >= trigger_end:
                    trigger_token = token
                    break
            
            if not trigger_token:
                continue
            
            # 提取主语（who）和宾语（whom）
            who = []
            whom = []
            when = ""
            where = []
            
            # 查找主语
            for token in trigger_sentence:
                # 如果token是触发词的主语
                if token.dep_ in ["nsubj", "nsubjpass"] and token.head == trigger_token:
                    # 提取完整的名词短语
                    subject_span = self._get_span_for_token(token)
                    subject_text = subject_span.text
                    
                    # 查找对应的实体
                    entity = self._find_entity_by_text(subject_text, entities)
                    if entity:
                        who.append({
                            "entity_id": entity["entity_id"],
                            "role": "AGENT"
                        })
            
            # 查找宾语
            for token in trigger_sentence:
                # 如果token是触发词的宾语
                if token.dep_ in ["dobj", "pobj", "attr"] and token.head == trigger_token:
                    # 提取完整的名词短语
                    object_span = self._get_span_for_token(token)
                    object_text = object_span.text
                    
                    # 查找对应的实体
                    entity = self._find_entity_by_text(object_text, entities)
                    if entity:
                        whom.append({
                            "entity_id": entity["entity_id"],
                            "role": "PATIENT"
                        })
            
            # 查找时间和地点
            for token in trigger_sentence:
                # 如果token是时间状语
                if token.dep_ in ["npadvmod", "advmod"] and token.head == trigger_token:
                    # 检查是否是时间实体
                    entity = self._find_entity_by_token(token, entities)
                    if entity and entity["type"] in ["TIME", "DATE"]:
                        when = entity["text"]
                
                # 如果token是地点状语
                if token.dep_ in ["pobj"] and token.head.dep_ == "prep" and token.head.head == trigger_token:
                    # 检查是否是地点实体
                    entity = self._find_entity_by_token(token, entities)
                    if entity and entity["type"] == "LOCATION":
                        where.append({
                            "entity_id": entity["entity_id"]
                        })
            
            # 创建事件
            event = {
                "event_id": f"EV{event_id_counter}",
                "type": trigger["potential_type"],
                "trigger": {
                    "trigger_id": trigger["trigger_id"],
                    "text": trigger["text"]
                },
                "elements": {
                    "who": who,
                    "whom": whom,
                    "when": when,
                    "where": where,
                    "why": "",  # 需要更复杂的分析
                    "how": ""   # 需要更复杂的分析
                },
                "source_text": trigger_sentence.text
            }
            
            events.append(event)
            event_id_counter += 1
        
        logger.info(f"提取到 {len(events)} 个事件基本要素")
        return events
    
    def _get_span_for_token(self, token):
        """获取token所在的完整名词短语"""
        # 如果token是名词短语的一部分，找到整个短语
        if token.dep_ in ["compound", "amod", "det", "nummod"]:
            # 找到短语的头部
            head = token.head
            while head.dep_ in ["compound", "amod", "det", "nummod"]:
                head = head.head
            
            # 收集短语的所有部分
            span_tokens = [head]
            for child in head.children:
                if child.dep_ in ["compound", "amod", "det", "nummod"]:
                    span_tokens.append(child)
            
            # 按照文本顺序排序
            span_tokens.sort(key=lambda t: t.i)
            
            # 创建span
            if len(span_tokens) > 1:
                return token.doc[span_tokens[0].i:span_tokens[-1].i + 1]
        
        # 如果token本身就是名词短语的头部，收集它的所有修饰语
        children = [token]
        for child in token.children:
            if child.dep_ in ["compound", "amod", "det", "nummod"]:
                children.append(child)
        
        # 按照文本顺序排序
        children.sort(key=lambda t: t.i)
        
        # 创建span
        if len(children) > 1:
            return token.doc[children[0].i:children[-1].i + 1]
        
        # 如果没有找到完整的短语，返回token本身
        return token.doc[token.i:token.i + 1]
    
    def _find_entity_by_text(self, text, entities):
        """根据文本查找实体"""
        # 规范化文本
        text = text.strip().lower()
        
        # 查找完全匹配的实体
        for entity in entities:
            if entity["text"].lower() == text:
                return entity
        
        # 查找部分匹配的实体
        for entity in entities:
            if text in entity["text"].lower() or entity["text"].lower() in text:
                return entity
        
        return None
    
    def _find_entity_by_token(self, token, entities):
        """根据token查找实体"""
        # 获取token的文本和位置
        token_text = token.text
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        # 查找包含token的实体
        for entity in entities:
            for mention in entity["mentions"]:
                mention_start = mention["position"][0]
                mention_end = mention["position"][1]
                
                # 检查token是否在实体提及的范围内
                if (mention_start <= token_start and mention_end >= token_end) or \
                   (token_start <= mention_start and token_end >= mention_end):
                    return entity
        
        return None 