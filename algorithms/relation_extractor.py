import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RelationExtractor:
    """关系提取器，负责提取实体间的关系和事件上下文"""
    
    def __init__(self):
        """初始化关系提取器"""
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
                logging.FileHandler(os.path.join(log_dir, "relation_extractor.log"), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def extract_relations(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取事件之间的关系
        
        Args:
            events: 事件列表
            
        Returns:
            带有关系信息的事件列表
        """
        logger.info("开始提取事件关系")
        
        # 如果事件数量少于2，无需提取关系
        if len(events) < 2:
            logger.info("事件数量少于2，无需提取关系")
            return events
        
        # 为每个事件添加关系字段
        for event in events:
            event["relations"] = []
        
        # 基于时间顺序的关系
        events_with_time = [e for e in events if e["elements"]["when"]]
        if events_with_time:
            # 按时间排序
            events_with_time.sort(key=lambda e: e["elements"]["when"])
            
            # 添加时序关系
            for i in range(len(events_with_time) - 1):
                current_event = events_with_time[i]
                next_event = events_with_time[i + 1]
                
                # 当前事件在下一个事件之前
                current_event["relations"].append({
                    "related_event_id": next_event["event_id"],
                    "relation_type": "BEFORE"
                })
                
                # 下一个事件在当前事件之后
                next_event["relations"].append({
                    "related_event_id": current_event["event_id"],
                    "relation_type": "AFTER"
                })
        
        # 基于共享实体的关系
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i == j:
                    continue
                
                # 检查是否共享主体
                event1_who_ids = [w["entity_id"] for w in event1["elements"]["who"]]
                event2_who_ids = [w["entity_id"] for w in event2["elements"]["who"]]
                
                shared_who = set(event1_who_ids).intersection(set(event2_who_ids))
                
                if shared_who:
                    # 添加共享主体关系
                    relation_exists = any(r["related_event_id"] == event2["event_id"] for r in event1["relations"])
                    if not relation_exists:
                        event1["relations"].append({
                            "related_event_id": event2["event_id"],
                            "relation_type": "SHARED_AGENT"
                        })
                
                # 检查是否存在主体-客体关系
                event1_whom_ids = [w["entity_id"] for w in event1["elements"]["whom"]]
                
                # 事件1的客体是事件2的主体
                shared_whom_who = set(event1_whom_ids).intersection(set(event2_who_ids))
                if shared_whom_who:
                    relation_exists = any(r["related_event_id"] == event2["event_id"] for r in event1["relations"])
                    if not relation_exists:
                        event1["relations"].append({
                            "related_event_id": event2["event_id"],
                            "relation_type": "OBJECT_TO_SUBJECT"
                        })
        
        logger.info("事件关系提取完成")
        return events 