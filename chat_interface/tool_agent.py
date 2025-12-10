"""
LLM4AD Chat Agent - 基于 LangChain 的工具调用 Agent
使用 LLM + Tools 的方式实现真正的对话式交互
"""

import os
import sys
import json
import http.client
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Tool(ABC):
    """工具基类"""
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def to_openai_function(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具"""
        pass


class ListMethodsTool(Tool):
    """列出可用方法的工具"""
    def __init__(self, config_manager):
        super().__init__(
            name="list_methods",
            description="列出所有可用的算法设计方法及其参数。当用户询问有哪些方法、想了解方法选项时调用此工具。",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.config_manager = config_manager
    
    def execute(self, **kwargs) -> str:
        methods = self.config_manager.get_available_methods()
        result = "## 可用的算法设计方法\n\n"
        for m in methods:
            info = self.config_manager.get_method_info(m)
            params = list(info.get('parameters', {}).keys())
            result += f"### {m}\n"
            result += f"{info.get('description', '')}\n"
            result += f"- 参数: {', '.join(params)}\n\n"
        return result


class ListTasksTool(Tool):
    """列出可用任务的工具"""
    def __init__(self, config_manager):
        super().__init__(
            name="list_tasks",
            description="列出所有可用的优化任务/问题，按类别分组。当用户询问有哪些任务、想了解可以解决什么问题时调用此工具。",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "任务类别：optimization（优化问题）、machine_learning（机器学习/强化学习）、science_discovery（科学发现）",
                        "enum": ["optimization", "machine_learning", "science_discovery"]
                    }
                },
                "required": []
            }
        )
        self.config_manager = config_manager
    
    def execute(self, category: str = None, **kwargs) -> str:
        tasks_by_cat = self.config_manager.get_tasks_by_category()
        
        categories = {
            "optimization": "优化问题",
            "machine_learning": "机器学习/强化学习",
            "science_discovery": "科学发现"
        }
        
        result = "## 可用的任务\n\n"
        for cat, cat_name in categories.items():
            if category and cat != category:
                continue
            cat_tasks = tasks_by_cat.get(cat, [])
            if cat_tasks:
                result += f"### {cat_name}\n"
                for t in cat_tasks:
                    info = self.config_manager.get_task_info(t)
                    params = list(info.get('parameters', {}).keys())
                    result += f"- **{t}**: {info.get('full_name', t)}\n"
                    if params:
                        result += f"  - 参数: {', '.join(params)}\n"
                result += "\n"
        return result


class GetMethodDetailsTool(Tool):
    """获取方法详情的工具"""
    def __init__(self, config_manager):
        super().__init__(
            name="get_method_details",
            description="获取指定算法设计方法的详细信息，包括完整描述和可配置参数。当用户想了解某个特定方法的详情时调用。",
            parameters={
                "type": "object",
                "properties": {
                    "method_name": {
                        "type": "string",
                        "description": "方法名称，如 EoH、FunSearch、HillClimb 等"
                    }
                },
                "required": ["method_name"]
            }
        )
        self.config_manager = config_manager
    
    def execute(self, method_name: str, **kwargs) -> str:
        info = self.config_manager.get_method_info(method_name)
        if not info:
            return f"未找到方法: {method_name}"
        
        result = f"## {info.get('full_name', method_name)} ({method_name})\n\n"
        result += f"{info.get('description', '')}\n\n"
        result += "### 可配置参数：\n"
        for param_name, param_info in info.get('parameters', {}).items():
            result += f"- **{param_name}** ({param_info.get('label', '')}): {param_info.get('help', '')}\n"
            result += f"  - 默认值: {param_info.get('default')}, 范围: [{param_info.get('min')}, {param_info.get('max')}]\n"
        return result


class SetConfigTool(Tool):
    """设置配置的工具"""
    
    # 任务名称纠正映射 - 将常见错误名称映射到正确的类名
    TASK_NAME_CORRECTIONS = {
        "TSPConstruct": "TSPEvaluation",
        "TSP": "TSPEvaluation",
        "tsp": "TSPEvaluation",
        "CVRPConstruct": "CVRPEvaluation",
        "CVRP": "CVRPEvaluation",
        "cvrp": "CVRPEvaluation",
        "KPConstruct": "KnapsackEvaluation",
        "KnapsackConstruct": "KnapsackEvaluation",
        "Knapsack": "KnapsackEvaluation",
        "knapsack": "KnapsackEvaluation",
        "OBP": "OBPEvaluation",
        "obp": "OBPEvaluation",
        "BinPacking": "OBPEvaluation",
        "OnlineBinPacking": "OBPEvaluation",
        "BP1DConstruct": "BP1DEvaluation",
        "BP1D": "BP1DEvaluation",
        "BP2DConstruct": "BP2DEvaluation",
        "BP2D": "BP2DEvaluation",
        "JSSPConstruct": "JSSPEvaluation",
        "JSSchedulingConstruct": "JSSPEvaluation",
        "JSSP": "JSSPEvaluation",
        "QAPConstruct": "QAPEvaluation",
        "QAP": "QAPEvaluation",
        "CarMountain": "CarMountainEvaluation",
        "Acrobot": "AcrobotEvaluation",
        "Pendulum": "PendulumEvaluation",
        "MoonLander": "MoonLanderEvaluation",
        "Feynman": "FeynmanEvaluation",
        "FeynmanSRSD": "FeynmanEvaluation",
    }
    
    def __init__(self, config_holder: Dict):
        super().__init__(
            name="set_config",
            description="设置算法设计的配置。可以同时设置方法、任务和多个参数。用户每次提到要修改配置时都应调用此工具。",
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "搜索方法：EoH, FunSearch, HillClimb, RandomSample, MCTS_AHD"
                    },
                    "task": {
                        "type": "string",
                        "description": "优化任务：OBPEvaluation(在线装箱), TSPEvaluation(TSP构造), CVRPEvaluation(CVRP), KnapsackEvaluation(背包)等"
                    },
                    "max_sample_nums": {
                        "type": "integer",
                        "description": "最大采样数量(所有方法通用)，默认50"
                    },
                    "num_samplers": {
                        "type": "integer",
                        "description": "并行采样器数量，默认2"
                    },
                    "num_evaluators": {
                        "type": "integer",
                        "description": "并行评估器数量，默认2"
                    },
                    "max_generations": {
                        "type": "integer",
                        "description": "最大代数(EoH专用)，默认10"
                    },
                    "pop_size": {
                        "type": "integer",
                        "description": "种群大小(EoH专用)，默认5"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "单次评估超时时间(秒)，默认20"
                    }
                },
                "required": []
            }
        )
        self.config_holder = config_holder
    
    def execute(self, method: str = None, task: str = None, **kwargs) -> str:
        updates = []
        if method:
            self.config_holder["method"] = method
            updates.append(f"方法: {method}")
        if task:
            # 尝试纠正任务名称
            corrected_task = self.TASK_NAME_CORRECTIONS.get(task, task)
            if corrected_task != task:
                updates.append(f"任务: {corrected_task} (已自动纠正自 {task})")
            else:
                updates.append(f"任务: {task}")
            self.config_holder["task"] = corrected_task
        
        # 更新参数 - 支持更多参数
        params = {}
        param_keys = ["max_sample_nums", "max_generations", "pop_size", 
                      "num_samplers", "num_evaluators", "timeout"]
        for key in param_keys:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
                updates.append(f"{key}: {kwargs[key]}")
        
        if params:
            self.config_holder["parameters"] = {**self.config_holder.get("parameters", {}), **params}
        
        if updates:
            result = f"✅ 配置已更新：\n- " + "\n- ".join(updates)
            # 显示当前完整状态
            m = self.config_holder.get("method")
            t = self.config_holder.get("task")
            if m and t:
                result += f"\n\n当前配置完整(方法:{m}, 任务:{t})，可以开始运行。"
            elif not m:
                result += "\n\n⚠️ 还需选择方法"
            elif not t:
                result += "\n\n⚠️ 还需选择任务"
            return result
        return "没有配置被更新"


class GetCurrentConfigTool(Tool):
    """获取当前配置的工具"""
    def __init__(self, config_holder: Dict):
        super().__init__(
            name="get_current_config",
            description="获取当前的配置状态，包括已选择的方法、任务和参数设置。",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.config_holder = config_holder
    
    def execute(self, **kwargs) -> str:
        method = self.config_holder.get("method")
        task = self.config_holder.get("task")
        params = self.config_holder.get("parameters", {})
        
        result = "当前配置：\n"
        result += f"- 方法: {method or '未选择'}\n"
        result += f"- 任务: {task or '未选择'}\n"
        if params:
            result += "- 参数:\n"
            for k, v in params.items():
                result += f"  - {k}: {v}\n"
        
        if method and task:
            result += "\n✅ 配置完成，可以开始运行算法设计。"
        else:
            missing = []
            if not method:
                missing.append("方法")
            if not task:
                missing.append("任务")
            result += f"\n⚠️ 还需要选择: {', '.join(missing)}"
        
        return result


class RunAlgorithmDesignTool(Tool):
    """运行算法设计的工具 - 这是核心！"""
    def __init__(self, config_holder: Dict, llm_config: Dict, run_callback: Callable = None):
        super().__init__(
            name="run_algorithm_design",
            description="启动算法设计。当用户说'开始'、'运行'、'启动'且方法和任务都已配置时，直接调用此工具(confirm=true)启动运行，无需再次询问用户确认。",
            parameters={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "设为true即可启动"
                    }
                },
                "required": []
            }
        )
        self.config_holder = config_holder
        self.llm_config = llm_config
        self.run_callback = run_callback
    
    def execute(self, confirm: bool = True, **kwargs) -> str:
        method = self.config_holder.get("method")
        task = self.config_holder.get("task")
        
        if not method:
            return "❌ 错误：请先选择一个方法（如 EoH、FunSearch）"
        if not task:
            return "❌ 错误：请先选择一个任务（如 OBPEvaluation、TSPEvaluation）"
        
        # 标记开始运行
        self.config_holder["_run_requested"] = True
        self.config_holder["_run_config"] = {
            "method": method,
            "task": task,
            "parameters": self.config_holder.get("parameters", {}),
            "llm_config": self.llm_config
        }
        
        params = self.config_holder.get("parameters", {})
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "默认参数"
        
        return f"🚀 启动算法设计！\n\n- 方法: {method}\n- 任务: {task}\n- 参数: {param_str}\n\n请查看下方实时输出..."


class ToolCallingAgent:
    """基于工具调用的对话 Agent"""
    
    SYSTEM_PROMPT = (
        "你是 LLM4AD 自动算法设计助手，一个专业的AI科研助手。\n\n"
        "【你的身份】\n"
        "我是 LLM4AD 智能助手，专门帮助研究人员使用大语言模型进行自动算法设计。"
        "我可以帮您选择合适的搜索方法（如 EoH、FunSearch）、配置优化任务（如装箱问题、TSP）、"
        "调整参数并启动算法设计过程。\n\n"
        "【工作流程】\n"
        "1. 理解用户需求，推荐合适的方法和任务\n"
        "2. 使用工具获取可用选项并设置配置\n"
        "3. 用户说'开始'、'运行'或确认后，立即调用 run_algorithm_design 启动\n\n"
        "【重要规则】\n"
        "- 当用户选择方法或任务时，立即用 set_config 记录\n"
        "- 用户可以一次修改多个参数，都要记录\n"
        "- 当用户说'开始'、'运行'、'启动'时，如果方法和任务都已配置，直接调用 run_algorithm_design(confirm=true) 启动，不要再询问确认\n"
        "- 保持专业简洁，像科研助手一样交流\n"
        "- 第一次对话时简短自我介绍\n"
    )
    
    def __init__(self, host: str, api_key: str, model: str = "gpt-4o-mini"):
        self.host = host
        self.api_key = api_key
        self.model = model
        self.timeout = 120
        
        # 配置持有者
        self.config_holder: Dict[str, Any] = {
            "method": None,
            "task": None,
            "parameters": {}
        }
        
        # 对话历史
        self.conversation_history: List[Dict] = []
        
        # 工具列表（稍后初始化）
        self.tools: List[Tool] = []
    
    def initialize_tools(self, config_manager, llm_config: Dict):
        """初始化工具"""
        self.tools = [
            ListMethodsTool(config_manager),
            ListTasksTool(config_manager),
            GetMethodDetailsTool(config_manager),
            SetConfigTool(self.config_holder),
            GetCurrentConfigTool(self.config_holder),
            RunAlgorithmDesignTool(self.config_holder, llm_config),
        ]
    
    def update_llm_config(self, host: str = None, api_key: str = None, model: str = None):
        """更新 LLM 配置"""
        if host:
            self.host = host
        if api_key:
            self.api_key = api_key
        if model:
            self.model = model
    
    @property
    def current_config(self) -> Dict:
        """获取当前配置 - 内外层 LLM 配置统一"""
        llm_cfg = {"host": self.host, "key": self.api_key, "model": self.model}
        return {
            "method": self.config_holder.get("method"),
            "task": self.config_holder.get("task"),
            "parameters": self.config_holder.get("parameters", {}),
            "llm": {
                "outer": llm_cfg,
                "inner": llm_cfg  # 内外层统一
            }
        }
    
    def _call_llm(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """调用 LLM API（支持工具调用）"""
        try:
            conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
            
            payload = {
                'model': self.model,
                'messages': messages,
                'max_tokens': 2048,
                'temperature': 0.7,
            }
            
            if tools:
                payload['tools'] = tools
                payload['tool_choice'] = 'auto'
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            conn.request('POST', '/v1/chat/completions', json.dumps(payload), headers)
            res = conn.getresponse()
            data = res.read().decode('utf-8')
            return json.loads(data)
            
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """执行工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(**arguments)
        return f"未找到工具: {tool_name}"
    
    def chat(self, user_message: str, config: Dict = None, config_manager = None) -> Dict[str, Any]:
        """处理用户消息
        
        Args:
            user_message: 用户输入的消息
            config: 外部传入的配置（用于同步）
            config_manager: 配置管理器（用于初始化工具）
        """
        # 如果传入了配置，同步到 config_holder
        if config:
            if config.get("method"):
                self.config_holder["method"] = config["method"]
            if config.get("task"):
                self.config_holder["task"] = config["task"]
            if config.get("parameters"):
                self.config_holder["parameters"] = config["parameters"]
            # 更新 LLM 配置
            llm_config = config.get("llm", {}).get("inner", {})
            if llm_config:
                self.update_llm_config(
                    host=llm_config.get("host"),
                    api_key=llm_config.get("key"),
                    model=llm_config.get("model")
                )
        
        # 初始化工具（如果需要）
        if config_manager and not self.tools:
            llm_config = config.get("llm", {}).get("inner", {}) if config else {}
            self.initialize_tools(config_manager, llm_config)
        
        # 添加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ] + self.conversation_history[-20:]  # 保留最近20条消息
        
        # 获取工具定义
        tools = [t.to_openai_function() for t in self.tools]
        
        # 调用 LLM
        response = self._call_llm(messages, tools)
        
        if "error" in response:
            return {
                "action": "chat",
                "message": f"调用 LLM 时出错: {response['error']}"
            }
        
        # 解析响应
        try:
            choice = response['choices'][0]
            message = choice['message']
            
            # 检查是否有工具调用
            if message.get('tool_calls'):
                tool_results = []
                for tool_call in message['tool_calls']:
                    func = tool_call['function']
                    tool_name = func['name']
                    
                    # 解析参数
                    try:
                        arguments = json.loads(func.get('arguments', '{}'))
                    except:
                        arguments = {}
                    
                    # 执行工具
                    result = self._execute_tool(tool_name, arguments)
                    tool_results.append({
                        "tool_call_id": tool_call['id'],
                        "name": tool_name,
                        "result": result
                    })
                
                # 将工具调用和结果添加到历史
                # 注意：当有 tool_calls 时，content 可能为 None，需要设为空字符串
                assistant_msg = dict(message)
                if assistant_msg.get('content') is None:
                    assistant_msg['content'] = ""
                self.conversation_history.append(assistant_msg)
                
                for tr in tool_results:
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["result"] or ""  # 确保不为 None
                    })
                
                # 再次调用 LLM 获取最终回复
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT}
                ] + self.conversation_history[-20:]
                
                final_response = self._call_llm(messages)
                
                if "error" not in final_response:
                    final_message = final_response['choices'][0]['message'].get('content') or ""
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_message
                    })
                    
                    # 构建工具调用描述
                    tool_info = "\n".join([f"🔧 调用了 `{tr['name']}`" for tr in tool_results])
                    
                    # 检查是否请求运行
                    run_requested = self.config_holder.get("_run_requested", False)
                    if run_requested:
                        self.config_holder["_run_requested"] = False
                        return {
                            "action": "run_algorithm",
                            "message": final_message,
                            "tool_calls": tool_info,
                            "config": self.config_holder.get("_run_config")
                        }
                    
                    return {
                        "action": "chat",
                        "message": final_message,
                        "tool_calls": tool_info
                    }
            
            # 没有工具调用，直接返回内容
            content = message.get('content') or ""
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            
            return {
                "action": "chat",
                "message": content
            }
            
        except Exception as e:
            return {
                "action": "chat",
                "message": f"解析响应时出错: {str(e)}"
            }
    
    def reset(self):
        """重置对话"""
        self.conversation_history = []
        self.config_holder = {
            "method": None,
            "task": None,
            "parameters": {}
        }
    
    def get_current_config(self) -> Dict:
        """获取当前配置"""
        return {
            "method": self.config_holder.get("method"),
            "task": self.config_holder.get("task"),
            "parameters": self.config_holder.get("parameters", {})
        }
