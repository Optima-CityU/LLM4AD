"""
LLM4AD Chat Agent - 外层对话 Agent
负责理解用户意图，调度任务，并与内层 LLM4AD 交互
"""

import json
import http.client
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ConversationContext:
    """对话上下文"""
    selected_method: Optional[str] = None
    selected_task: Optional[str] = None
    parameters: Dict[str, Any] = None
    history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.history is None:
            self.history = []


# System prompt as a constant
SYSTEM_PROMPT_TEXT = (
    "你是 LLM4AD 自动算法设计助手，一个专业的科研助手。"
    "你的任务是帮助用户使用 LLM4AD 框架进行自动算法设计。\n\n"
    "## 你的能力：\n"
    "1. 帮助用户选择合适的算法设计方法（如 EoH、FunSearch、HillClimb、RandSample 等）\n"
    "2. 帮助用户选择要解决的优化任务（如在线装箱问题、TSP、车辆路径问题等）\n"
    "3. 帮助用户配置参数（采样数量、种群大小、迭代次数等）\n"
    "4. 启动算法设计过程并展示结果\n\n"
    "## 可用的方法（Methods）：\n"
    "- EoH (Evolution of Heuristics): 启发式演化方法，通过交叉变异操作进化启发式算法，适合大规模搜索\n"
    "- FunSearch: 函数搜索方法，Google DeepMind 提出，适合发现新颖的算法\n"
    "- HillClimb: 爬山算法，简单有效的局部搜索方法\n"
    "- RandSample: 随机采样方法，作为基准方法使用\n"
    "- MoEaD: 多目标演化算法\n"
    "- NSGA2: 非支配排序遗传算法\n"
    "- ReEvo: 反射演化方法\n"
    "- MEoH: 多专家启发式演化\n\n"
    "## 可用的任务（Tasks）：\n"
    "### 优化问题：\n"
    "- OBPEvaluation (Online Bin Packing): 在线装箱问题\n"
    "- TSPEvaluation: 旅行商问题（构造式）\n"
    "- CVRPEvaluation: 有容量限制的车辆路径问题\n"
    "- KnapsackEvaluation: 背包问题\n"
    "- JSSPEvaluation: 作业车间调度问题\n"
    "- BP1DEvaluation: 一维装箱问题\n"
    "- BP2DEvaluation: 二维装箱问题\n"
    "- QAPEvaluation: 二次分配问题\n\n"
    "### 机器学习问题：\n"
    "- CarMountainEvaluation: 山地车控制问题\n"
    "- AcrobotEvaluation: 双摆控制问题\n"
    "- PendulumEvaluation: 倒立摆控制问题\n\n"
    "### 科学发现：\n"
    "- FeynmanEvaluation: 符号回归科学发现\n\n"
    "## 回复格式：\n"
    "你需要以 JSON 格式返回你的决策，格式如下：\n"
    '{"action": "action_type", "message": "给用户的回复消息", '
    '"config": {"method": "方法名（如果有）", "task": "任务名（如果有）", "parameters": {}}}\n\n'
    "action 类型包括：\n"
    "- chat: 普通对话，不需要执行任何操作\n"
    "- update_config: 更新配置（方法、任务或参数）\n"
    "- show_config: 显示配置面板让用户手动设置\n"
    "- run_algorithm: 开始运行算法设计\n"
    "- confirm: 需要用户确认操作\n\n"
    "## 重要规则：\n"
    "1. 在用户明确指定方法和任务后才能运行算法\n"
    "2. 如果用户描述的问题不清楚，主动询问\n"
    "3. 推荐时解释为什么推荐这个方法/任务\n"
    "4. 始终保持专业、简洁的科研风格\n"
    "5. 如果用户想了解更多信息，提供详细解释\n\n"
    "## 参数说明（供你参考和解释）：\n"
    "- max_sample_nums: 最大采样次数，决定搜索规模\n"
    "- max_generations: 最大代数，决定演化轮数\n"
    "- pop_size: 种群大小，影响多样性\n"
    "- num_samplers: 采样器数量，影响并行度\n"
    "- num_evaluators: 评估器数量，影响评估速度\n"
)


class ChatAgent:
    """外层对话 Agent，负责理解用户意图并调度任务"""
    
    SYSTEM_PROMPT = SYSTEM_PROMPT_TEXT

    def __init__(self, host: str, api_key: str, model: str = "gpt-4o-mini"):
        self.host = host
        self.api_key = api_key
        self.model = model
        self.context = ConversationContext()
        self.timeout = 60
    
    def update_config(self, host: str = None, api_key: str = None, model: str = None):
        """更新 LLM 配置"""
        if host:
            self.host = host
        if api_key:
            self.api_key = api_key
        if model:
            self.model = model
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用 LLM API"""
        try:
            conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
            payload = json.dumps({
                'max_tokens': 2048,
                'temperature': 0.7,
                'model': self.model,
                'messages': messages
            })
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            conn.request('POST', '/v1/chat/completions', payload, headers)
            res = conn.getresponse()
            data = res.read().decode('utf-8')
            data = json.loads(data)
            return data['choices'][0]['message']['content']
        except Exception as e:
            return json.dumps({
                "action": "chat",
                "message": f"抱歉，与 AI 服务通信时出现问题：{str(e)}。请检查您的 API 配置。"
            })
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # 尝试直接解析
                json_str = response.strip()
            
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            # 如果解析失败，返回普通对话
            return {
                "action": "chat",
                "message": response
            }
    
    def _build_context_message(self, current_config: Dict[str, Any]) -> str:
        """构建当前上下文信息"""
        context_parts = []
        
        if current_config.get("method"):
            context_parts.append(f"当前选择的方法: {current_config['method']}")
        else:
            context_parts.append("方法: 未选择")
            
        if current_config.get("task"):
            context_parts.append(f"当前选择的任务: {current_config['task']}")
        else:
            context_parts.append("任务: 未选择")
        
        if current_config.get("parameters"):
            params_str = ", ".join([f"{k}={v}" for k, v in current_config["parameters"].items()])
            context_parts.append(f"参数配置: {params_str}")
        
        return "\n".join(context_parts)
    
    def chat(self, user_message: str, current_config: Dict[str, Any], config_manager) -> Dict[str, Any]:
        """处理用户消息并返回响应"""
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # 添加上下文信息
        context_info = self._build_context_message(current_config)
        messages.append({
            "role": "system", 
            "content": f"当前配置状态:\n{context_info}"
        })
        
        # 添加历史对话（最近5轮）
        for msg in self.context.history[-10:]:
            messages.append(msg)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 调用 LLM
        response = self._call_llm(messages)
        
        # 解析响应
        result = self._parse_response(response)
        
        # 更新对话历史
        self.context.history.append({"role": "user", "content": user_message})
        self.context.history.append({"role": "assistant", "content": result.get("message", response)})
        
        # 处理特殊动作
        action = result.get("action", "chat")
        
        if action == "run_algorithm":
            # 检查是否有足够的配置
            config = result.get("config", {})
            method = config.get("method") or current_config.get("method")
            task = config.get("task") or current_config.get("task")
            
            if not method or not task:
                return {
                    "action": "chat",
                    "message": "⚠️ 在运行算法之前，需要先选择方法和任务。\n\n" + 
                              f"当前状态：\n- 方法: {method or '未选择'}\n- 任务: {task or '未选择'}\n\n" +
                              "请告诉我您想使用什么方法解决什么问题？"
                }
            
            # 合并配置
            merged_config = {
                "method": method,
                "task": task,
                "parameters": {**current_config.get("parameters", {}), **config.get("parameters", {})}
            }
            
            # 设置默认参数
            if not merged_config["parameters"]:
                merged_config["parameters"] = config_manager.get_default_parameters(method)
            
            return {
                "action": "run_algorithm",
                "message": result.get("message", f"好的，我将使用 {method} 方法为您设计 {task} 问题的算法..."),
                "config": merged_config
            }
        
        elif action == "update_config":
            config = result.get("config", {})
            return {
                "action": "update_config",
                "message": result.get("message", "配置已更新。"),
                "config": config
            }
        
        elif action == "show_config":
            return {
                "action": "show_config",
                "message": result.get("message", "请在配置面板中进行设置。"),
                "config_data": {
                    "methods": config_manager.get_available_methods(),
                    "tasks": config_manager.get_available_tasks(),
                    "current_method": current_config.get("method"),
                    "current_task": current_config.get("task")
                }
            }
        
        else:
            return {
                "action": "chat",
                "message": result.get("message", response)
            }
    
    def reset_context(self):
        """重置对话上下文"""
        self.context = ConversationContext()


class StreamingChatAgent(ChatAgent):
    """支持流式输出的 Chat Agent"""
    
    def chat_stream(self, user_message: str, current_config: Dict[str, Any], config_manager):
        """流式处理用户消息"""
        # 构建消息列表
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        context_info = self._build_context_message(current_config)
        messages.append({
            "role": "system", 
            "content": f"当前配置状态:\n{context_info}"
        })
        
        for msg in self.context.history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_message})
        
        # 流式调用 (简化版，实际可以使用 SSE)
        full_response = self._call_llm(messages)
        
        # 模拟流式输出
        for char in full_response:
            yield char
        
        # 更新历史
        result = self._parse_response(full_response)
        self.context.history.append({"role": "user", "content": user_message})
        self.context.history.append({"role": "assistant", "content": result.get("message", full_response)})
