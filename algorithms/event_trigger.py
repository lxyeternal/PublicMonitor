import logging
import os
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EventTriggerExtractor:
    """事件触发词提取器，负责从文本中识别事件触发词"""
    
    def __init__(self):
        """初始化事件触发词提取器"""
        self.setup_logging()
        self.load_trigger_words()
        
    def setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "event_trigger.log"), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_trigger_words(self):
        """加载常见事件触发词列表"""
        # 这里是一些常见的事件触发词，实际应用中可以从文件或数据库加载更完整的列表
        self.trigger_words = {
            "STATEMENT": ["say", "announce", "state", "declare", "claim", "report", "mention", "tell", "speak", "assert"],
            "MOVEMENT": ["go", "move", "travel", "arrive", "leave", "depart", "return", "enter", "exit", "flee"],
            "TRANSACTION": ["buy", "sell", "trade", "purchase", "acquire", "pay", "spend", "invest", "donate", "fund"],
            "CONFLICT": ["attack", "fight", "war", "battle", "strike", "bomb", "shoot", "kill", "destroy", "defeat"],
            "BUSINESS": ["merge", "acquire", "launch", "start", "found", "establish", "expand", "grow", "develop", "build"],
            "JUSTICE": ["arrest", "charge", "convict", "sentence", "sue", "prosecute", "investigate", "trial", "judge", "rule"],
            "LIFE": ["born", "die", "marry", "divorce", "graduate", "study", "educate", "live", "grow", "age"],
            "CONTACT": ["meet", "visit", "contact", "call", "email", "write", "talk", "discuss", "negotiate", "consult"],
            "PERSONNEL": ["hire", "fire", "resign", "appoint", "elect", "nominate", "promote", "demote", "employ", "work"]
        }
        
        # 扁平化触发词列表，用于快速查找
        self.all_triggers = {}
        for event_type, words in self.trigger_words.items():
            for word in words:
                self.all_triggers[word] = event_type
                # 添加过去式和现在进行时形式
                if word.endswith('e'):
                    self.all_triggers[word + 'd'] = event_type
                    self.all_triggers[word[:-1] + 'ing'] = event_type
                else:
                    self.all_triggers[word + 'ed'] = event_type
                    self.all_triggers[word + 'ing'] = event_type
        
        logger.info(f"加载了 {len(self.all_triggers)} 个事件触发词")
    
    def extract_triggers(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取事件触发词
        
        Args:
            text: 输入文本
            
        Returns:
            触发词列表，每个触发词包含ID、文本、位置和可能的事件类型
        """
        logger.info("开始提取事件触发词")
        
        # 分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 提取触发词
        triggers = []
        trigger_id_counter = 1
        
        for word in words:
            if word in self.all_triggers:
                # 查找单词在原文中的位置
                pattern = r'\b' + re.escape(word) + r'\b'
                for match in re.finditer(pattern, text.lower()):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # 获取原文中的实际文本（保留大小写）
                    original_text = text[start_pos:end_pos]
                    
                    trigger = {
                        "trigger_id": f"T{trigger_id_counter}",
                        "text": original_text,
                        "position": [start_pos, end_pos],
                        "potential_type": self.all_triggers[word]
                    }
                    triggers.append(trigger)
                    trigger_id_counter += 1
        
        logger.info(f"提取到 {len(triggers)} 个事件触发词")
        return triggers 