import os
import json
import logging
import uuid
from typing import Dict, List, Any

from services.llm_service import LLMService
from services.text_processor import TextProcessor
from algorithms.ner_extractor import NERExtractor
from algorithms.event_trigger import EventTriggerExtractor
from algorithms.srl_extractor import SRLExtractor
from algorithms.relation_extractor import RelationExtractor

logger = logging.getLogger(__name__)

class EventExtractor:
    """事件提取服务，负责从文本中提取事件结构"""
    
    def __init__(self):
        """初始化事件提取器"""
        self.llm_service = LLMService()
        self.text_processor = TextProcessor()
        self.ner_extractor = NERExtractor()
        self.trigger_extractor = EventTriggerExtractor()
        self.srl_extractor = SRLExtractor()
        self.relation_extractor = RelationExtractor()
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
                logging.FileHandler(os.path.join(log_dir, "event_extractor.log"), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def extract_entities_and_triggers(self, text):
        """
        使用传统NLP方法提取实体和事件触发词
        
        Args:
            text: 预处理后的文本
            
        Returns:
            实体和触发词信息
        """
        logger.info("开始使用传统NLP方法提取实体和事件触发词")
        
        # 使用NER提取实体
        entities = self.ner_extractor.extract_entities(text)
        
        # 使用触发词提取器提取事件触发词
        triggers = self.trigger_extractor.extract_triggers(text)
        
        # 整合结果
        result = {
            "entities": entities,
            "event_triggers": triggers
        }
        
        logger.info(f"传统方法成功提取 {len(entities)} 个实体和 {len(triggers)} 个事件触发词")
        
        # 使用LLM补充和优化提取结果
        enhanced_result = self.enhance_extraction_with_llm(text, result)
        
        return enhanced_result
    
    def enhance_extraction_with_llm(self, text, extraction_result):
        """
        使用LLM补充和优化传统方法的提取结果
        
        Args:
            text: 预处理后的文本
            extraction_result: 传统方法提取的结果
            
        Returns:
            增强后的提取结果
        """
        logger.info("开始使用LLM补充和优化提取结果")
        
        # 加载提示词模板
        prompt_template = self.text_processor.load_prompt("entity_extraction")
        
        # 使用f-string而不是format方法
        prompt = f"{prompt_template}".replace("{text}", text)
        
        # 构建消息
        messages = [
            {"role": "system", "content": "You are an expert in entity and event trigger extraction."},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM服务
        response = self.llm_service.query(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # 解析响应
        try:
            llm_result = json.loads(response)
            logger.info(f"LLM成功提取 {len(llm_result.get('entities', []))} 个实体和 {len(llm_result.get('event_triggers', []))} 个事件触发词")
            
            # 合并传统方法和LLM的结果
            merged_result = self.merge_extraction_results(extraction_result, llm_result)
            logger.info(f"合并后共有 {len(merged_result.get('entities', []))} 个实体和 {len(merged_result.get('event_triggers', []))} 个事件触发词")
            
            return merged_result
        except Exception as e:
            logger.error(f"解析LLM提取结果失败: {e}")
            logger.error(f"原始响应: {response}")
            return extraction_result
    
    def merge_extraction_results(self, traditional_result, llm_result):
        """
        合并传统方法和LLM的提取结果
        
        Args:
            traditional_result: 传统方法的提取结果
            llm_result: LLM的提取结果
            
        Returns:
            合并后的结果
        """
        logger.info("开始合并传统方法和LLM的提取结果")
        
        # 获取传统方法的实体和触发词
        traditional_entities = traditional_result.get("entities", [])
        traditional_triggers = traditional_result.get("event_triggers", [])
        
        # 获取LLM的实体和触发词
        llm_entities = llm_result.get("entities", [])
        llm_triggers = llm_result.get("event_triggers", [])
        
        # 合并实体
        merged_entities = traditional_entities.copy()
        entity_texts = [e["text"].lower() for e in merged_entities]
        
        # 添加LLM提取的新实体
        entity_id_counter = len(merged_entities) + 1
        for llm_entity in llm_entities:
            if llm_entity["text"].lower() not in entity_texts:
                # 更新实体ID
                llm_entity["entity_id"] = f"E{entity_id_counter}"
                merged_entities.append(llm_entity)
                entity_texts.append(llm_entity["text"].lower())
                entity_id_counter += 1
        
        # 合并触发词
        merged_triggers = traditional_triggers.copy()
        trigger_texts = [(t["text"].lower(), t["position"][0], t["position"][1]) for t in merged_triggers]
        
        # 添加LLM提取的新触发词
        trigger_id_counter = len(merged_triggers) + 1
        for llm_trigger in llm_triggers:
            trigger_key = (llm_trigger["text"].lower(), llm_trigger["position"][0], llm_trigger["position"][1])
            if trigger_key not in trigger_texts:
                # 更新触发词ID
                llm_trigger["trigger_id"] = f"T{trigger_id_counter}"
                merged_triggers.append(llm_trigger)
                trigger_texts.append(trigger_key)
                trigger_id_counter += 1
        
        # 返回合并后的结果
        merged_result = {
            "entities": merged_entities,
            "event_triggers": merged_triggers
        }
        
        return merged_result
    
    def construct_events(self, text, entities, triggers):
        """
        构建事件结构
        
        Args:
            text: 预处理后的文本
            entities: 提取的实体信息
            triggers: 提取的触发词信息
            
        Returns:
            事件结构
        """
        logger.info("开始构建事件结构")
        
        # 使用SRL提取事件基本要素
        basic_events = self.srl_extractor.extract_srl(text, triggers, entities)
        
        # 使用LLM补充和优化事件结构
        enhanced_events = self.enhance_events_with_llm(text, basic_events, entities, triggers)
        
        return {"events": enhanced_events}
    
    def enhance_events_with_llm(self, text, basic_events, entities, triggers):
        """
        使用LLM补充和优化事件结构
        
        Args:
            text: 预处理后的文本
            basic_events: 基本事件结构
            entities: 提取的实体信息
            triggers: 提取的触发词信息
            
        Returns:
            增强后的事件结构
        """
        logger.info("开始使用LLM补充和优化事件结构")
        
        # 加载提示词模板
        prompt_template = self.text_processor.load_prompt("event_construction")
        
        # 使用安全的替换方法而不是format
        prompt = prompt_template
        prompt = prompt.replace("{text}", text)
        prompt = prompt.replace("{entities}", json.dumps(entities, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{triggers}", json.dumps(triggers, ensure_ascii=False, indent=2))
        
        # 构建消息
        messages = [
            {"role": "system", "content": "You are an expert in event analysis and construction."},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM服务
        response = self.llm_service.query(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # 解析响应
        try:
            llm_result = json.loads(response)
            llm_events = llm_result.get("events", [])
            logger.info(f"LLM成功构建 {len(llm_events)} 个事件")
            
            # 合并基本事件和LLM构建的事件
            merged_events = self.merge_events(basic_events, llm_events)
            logger.info(f"合并后共有 {len(merged_events)} 个事件")
            
            return merged_events
        except Exception as e:
            logger.error(f"解析LLM事件构建结果失败: {e}")
            logger.error(f"原始响应: {response}")
            return basic_events
    
    def merge_events(self, basic_events, llm_events):
        """
        合并基本事件和LLM构建的事件
        
        Args:
            basic_events: 基本事件结构
            llm_events: LLM构建的事件结构
            
        Returns:
            合并后的事件结构
        """
        logger.info("开始合并基本事件和LLM构建的事件")
        
        # 创建事件ID到事件的映射
        basic_event_map = {event["event_id"]: event for event in basic_events}
        
        # 合并事件
        merged_events = basic_events.copy()
        
        # 添加LLM构建的新事件
        event_id_counter = len(merged_events) + 1
        for llm_event in llm_events:
            # 检查是否有匹配的基本事件
            matched = False
            for basic_event in basic_events:
                # 如果触发词相同，认为是同一事件
                if (basic_event["trigger"]["text"].lower() == llm_event["trigger"]["text"].lower()):
                    # 更新基本事件的信息
                    basic_event_map[basic_event["event_id"]].update({
                        "summary": llm_event.get("summary", ""),
                        "type": llm_event.get("type", basic_event["type"]),
                        "sentiment": llm_event.get("sentiment", {"polarity": "NEUTRAL", "intensity": 0.5}),
                        "importance": llm_event.get("importance", 3),
                        "confidence": llm_event.get("confidence", 0.8)
                    })
                    
                    # 补充事件要素
                    for key in ["why", "how"]:
                        if key in llm_event["elements"] and llm_event["elements"][key]:
                            basic_event_map[basic_event["event_id"]]["elements"][key] = llm_event["elements"][key]
                    
                    matched = True
                    break
            
            # 如果没有匹配的基本事件，添加为新事件
            if not matched:
                # 更新事件ID
                llm_event["event_id"] = f"EV{event_id_counter}"
                merged_events.append(llm_event)
                event_id_counter += 1
        
        return merged_events
    
    def integrate_events(self, text, events, entities, document_id=None):
        """
        整合事件结构
        
        Args:
            text: 预处理后的文本
            events: 初步构建的事件
            entities: 提取的实体信息
            document_id: 文档ID
            
        Returns:
            整合后的事件结构
        """
        logger.info("开始整合事件结构")
        
        # 提取事件关系
        events_with_relations = self.relation_extractor.extract_relations(events)
        
        # 使用LLM进行最终整合
        final_result = self.final_integration_with_llm(text, events_with_relations, entities, document_id)
        
        return final_result
    
    def final_integration_with_llm(self, text, events, entities, document_id):
        """
        使用LLM进行最终整合
        
        Args:
            text: 预处理后的文本
            events: 带有关系的事件
            entities: 提取的实体信息
            document_id: 文档ID
            
        Returns:
            最终整合的结果
        """
        logger.info("开始使用LLM进行最终整合")
        
        # 加载提示词模板
        prompt_template = self.text_processor.load_prompt("event_integration")
        
        # 使用安全的替换方法而不是format
        prompt = prompt_template
        prompt = prompt.replace("{text}", text)
        prompt = prompt.replace("{events}", json.dumps(events, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{entities}", json.dumps(entities, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{document_id}", str(document_id) if document_id else "")
        
        # 构建消息
        messages = [
            {"role": "system", "content": "You are an expert in event integration and quality control."},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM服务
        response = self.llm_service.query(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # 解析响应
        try:
            final_result = json.loads(response)
            logger.info(f"LLM成功整合 {len(final_result.get('events', []))} 个事件")
            return final_result
        except Exception as e:
            logger.error(f"解析LLM事件整合结果失败: {e}")
            logger.error(f"原始响应: {response}")
            
            # 如果LLM整合失败，返回基本整合结果
            basic_result = {
                "document_id": document_id,
                "events": events,
                "entities": entities
            }
            return basic_result
    
    def extract_events_from_text(self, text, document_id=None):
        """
        从文本中提取事件结构的主流程
        
        Args:
            text: 原始文本
            document_id: 文档ID
            
        Returns:
            完整的事件结构
        """
        logger.info("开始从文本中提取事件结构")
        
        # 文本预处理
        processed_text = self.text_processor.preprocess_text(text)
        
        # 步骤1: 提取实体和触发词
        extraction_result = self.extract_entities_and_triggers(processed_text)
        entities = extraction_result.get("entities", [])
        triggers = extraction_result.get("event_triggers", [])
        
        # 步骤2: 构建事件结构
        construction_result = self.construct_events(processed_text, entities, triggers)
        events = construction_result.get("events", [])
        
        # 步骤3: 整合事件结构
        final_result = self.integrate_events(processed_text, events, entities, document_id)
        
        logger.info("事件结构提取完成")
        return final_result
    
    def extract_events_from_file(self, file_path, document_id=None):
        """
        从文件中提取事件结构
        
        Args:
            file_path: 文本文件路径
            document_id: 文档ID
            
        Returns:
            完整的事件结构
        """
        logger.info(f"开始从文件中提取事件结构: {file_path}")
        
        # 读取文件
        text = self.text_processor.read_text_file(file_path)
        if text is None:
            logger.error("文件读取失败，无法提取事件")
            return None
        
        # 如果未提供文档ID，使用文件名作为文档ID
        if document_id is None:
            document_id = os.path.basename(file_path)
        
        # 提取事件结构
        result = self.extract_events_from_text(text, document_id)
        
        # 保存结果到JSON文件
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{document_id}_events.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"事件结构已保存到: {output_file}")
        return result 