"""
LLM4AD Config Manager - 配置管理器
自动从 llm4ad/method 和 llm4ad/task 中抓取可用的方法和任务
"""

import os
import sys
import inspect
from typing import Dict, List, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfigManager:
    """配置管理器，自动扫描 LLM4AD 的方法和任务"""
    
    # 方法的额外描述信息（可选，用于增强显示）
    METHOD_DESCRIPTIONS = {
        "EoH": "启发式演化方法，通过交叉变异操作进化启发式算法",
        "FunSearch": "Google DeepMind 提出的函数搜索方法，适合发现新颖算法",
        "HillClimb": "爬山算法，通过局部搜索不断改进解",
        "RandSample": "随机采样方法，作为基准对比",
        "MCTS_AHD": "蒙特卡洛树搜索自动启发式设计",
        "MEoH": "多专家启发式演化，使用多个LLM专家",
        "MOEAD": "基于分解的多目标演化算法",
        "NSGA2": "非支配排序遗传算法，适用于多目标优化",
        "ReEvo": "反思演化方法，通过自我反思改进",
        "LHNS": "大邻域搜索启发式方法",
    }
    
    # 任务分类映射
    TASK_CATEGORIES = {
        "OBP": ("在线装箱问题", "optimization"),
        "TSP": ("旅行商问题", "optimization"),
        "CVRP": ("有容量车辆路径问题", "optimization"),
        "BP1D": ("一维装箱问题", "optimization"),
        "BP2D": ("二维装箱问题", "optimization"),
        "JSSP": ("作业车间调度问题", "optimization"),
        "CFLP": ("设施选址问题", "optimization"),
        "Knapsack": ("背包问题", "optimization"),
        "CarMountain": ("小车爬山（强化学习）", "machine_learning"),
        "Acrobot": ("杂技机器人（强化学习）", "machine_learning"),
        "Feynman": ("符号回归", "science_discovery"),
        "BG": ("博弈论", "optimization"),
        "ASP": ("答案集编程", "optimization"),
    }
    
    # 通用参数模板
    COMMON_PARAMETERS = {
        "max_sample_nums": {
            "type": "int",
            "default": 50,
            "min": 5,
            "max": 10000,
            "label": "最大采样数",
            "help": "搜索的最大样本数量"
        },
        "num_samplers": {
            "type": "int",
            "default": 2,
            "min": 1,
            "max": 16,
            "label": "采样器数量",
            "help": "并行采样的线程数"
        },
        "num_evaluators": {
            "type": "int",
            "default": 2,
            "min": 1,
            "max": 16,
            "label": "评估器数量",
            "help": "并行评估的线程数"
        },
    }
    
    # 演化方法特有参数
    EVOLUTION_PARAMETERS = {
        "max_generations": {
            "type": "int",
            "default": 10,
            "min": 1,
            "max": 100,
            "label": "最大代数",
            "help": "演化的最大轮数"
        },
        "pop_size": {
            "type": "int",
            "default": 5,
            "min": 2,
            "max": 100,
            "label": "种群大小",
            "help": "每代保留的个体数量"
        },
    }
    
    def __init__(self):
        """初始化配置管理器，自动扫描可用的方法和任务"""
        self._methods: Dict[str, Dict] = {}
        self._tasks: Dict[str, Dict] = {}
        self._loaded = False
    
    def _ensure_loaded(self):
        """确保已加载方法和任务"""
        if not self._loaded:
            self._load_methods()
            self._load_tasks()
            self._loaded = True
    
    def _load_methods(self):
        """从 llm4ad.method 加载所有方法"""
        try:
            # 清理环境变量避免库冲突
            env_backup = {}
            for key in ['DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH']:
                if key in os.environ:
                    env_backup[key] = os.environ.pop(key)
            
            try:
                import llm4ad
                from llm4ad.method import import_all_method_classes_from_subfolders
                
                llm4ad_path = os.path.dirname(llm4ad.__file__)
                import_all_method_classes_from_subfolders(os.path.join(llm4ad_path, 'method'))
                
                # 获取所有方法类
                for name, obj in vars(llm4ad.method).items():
                    if inspect.isclass(obj) and hasattr(obj, 'run') and not name.startswith('_'):
                        # 判断是否是演化方法
                        is_evolution = name in ['EoH', 'MEoH', 'MOEAD', 'NSGA2', 'ReEvo']
                        
                        # 构建参数
                        params = dict(self.COMMON_PARAMETERS)
                        if is_evolution:
                            params.update(self.EVOLUTION_PARAMETERS)
                        
                        self._methods[name] = {
                            "name": name,
                            "full_name": name,
                            "description": self.METHOD_DESCRIPTIONS.get(name, f"{name} 方法"),
                            "class_name": name,
                            "parameters": params
                        }
            finally:
                # 恢复环境变量
                for key, val in env_backup.items():
                    os.environ[key] = val
                    
        except Exception as e:
            print(f"加载方法时出错: {e}")
            # 使用默认方法列表
            self._methods = {
                "EoH": {"name": "EoH", "full_name": "Evolution of Heuristics", 
                        "description": "启发式演化方法", "class_name": "EoH",
                        "parameters": {**self.COMMON_PARAMETERS, **self.EVOLUTION_PARAMETERS}},
                "FunSearch": {"name": "FunSearch", "full_name": "Function Search",
                              "description": "函数搜索方法", "class_name": "FunSearch",
                              "parameters": self.COMMON_PARAMETERS},
            }
    
    def _load_tasks(self):
        """从 llm4ad.task 加载所有任务"""
        try:
            # 清理环境变量
            env_backup = {}
            for key in ['DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH']:
                if key in os.environ:
                    env_backup[key] = os.environ.pop(key)
            
            try:
                import llm4ad
                from llm4ad.task import import_all_evaluation_classes
                
                llm4ad_path = os.path.dirname(llm4ad.__file__)
                import_all_evaluation_classes(os.path.join(llm4ad_path, 'task'))
                
                # 获取所有任务类
                for name, obj in vars(llm4ad.task).items():
                    if inspect.isclass(obj) and 'Evaluation' in name and not name.startswith('_'):
                        # 推断任务类别和描述
                        category = "optimization"
                        description = name
                        
                        for prefix, (desc, cat) in self.TASK_CATEGORIES.items():
                            if prefix in name:
                                description = desc
                                category = cat
                                break
                        
                        # 处理 CB (COBench) 后缀
                        if name.endswith('CB'):
                            description += " (CO-Bench)"
                        
                        self._tasks[name] = {
                            "name": name,
                            "full_name": description,
                            "description": description,
                            "category": category,
                            "class_name": name,
                            "parameters": {
                                "timeout_seconds": {
                                    "type": "int",
                                    "default": 30,
                                    "min": 5,
                                    "max": 300,
                                    "label": "超时时间(秒)",
                                    "help": "单次评估的最大时间"
                                }
                            }
                        }
            finally:
                for key, val in env_backup.items():
                    os.environ[key] = val
                    
        except Exception as e:
            print(f"加载任务时出错: {e}")
            # 使用默认任务列表
            self._tasks = {
                "OBPEvaluation": {"name": "OBPEvaluation", "full_name": "在线装箱问题",
                                  "description": "在线装箱问题", "category": "optimization",
                                  "class_name": "OBPEvaluation", "parameters": {}},
                "TSPEvaluation": {"name": "TSPEvaluation", "full_name": "旅行商问题",
                                  "description": "旅行商问题", "category": "optimization",
                                  "class_name": "TSPEvaluation", "parameters": {}},
            }
    
    def get_available_methods(self) -> List[str]:
        """获取所有可用方法的名称列表"""
        self._ensure_loaded()
        return list(self._methods.keys())
    
    def get_available_tasks(self) -> List[str]:
        """获取所有可用任务的名称列表"""
        self._ensure_loaded()
        return list(self._tasks.keys())
    
    def get_method_info(self, method_name: str) -> Dict[str, Any]:
        """获取方法的详细信息"""
        self._ensure_loaded()
        return self._methods.get(method_name, {})
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """获取任务的详细信息"""
        self._ensure_loaded()
        return self._tasks.get(task_name, {})
    
    def get_method_description(self, method_name: str) -> str:
        """获取方法的描述"""
        info = self.get_method_info(method_name)
        return info.get("description", "")
    
    def get_task_description(self, task_name: str) -> str:
        """获取任务的描述"""
        info = self.get_task_info(task_name)
        return info.get("description", "")
    
    def get_method_parameters(self, method_name: str) -> Dict[str, Any]:
        """获取方法的参数配置"""
        info = self.get_method_info(method_name)
        return info.get("parameters", {})
    
    def get_task_parameters(self, task_name: str) -> Dict[str, Any]:
        """获取任务的参数配置"""
        info = self.get_task_info(task_name)
        return info.get("parameters", {})
    
    def get_tasks_by_category(self, category: str) -> List[str]:
        """按类别获取任务列表"""
        self._ensure_loaded()
        return [name for name, info in self._tasks.items() 
                if info.get("category") == category]
    
    def get_all_categories(self) -> List[str]:
        """获取所有任务类别"""
        self._ensure_loaded()
        categories = set(info.get("category", "other") for info in self._tasks.values())
        return list(categories)
    
    def get_method_class_name(self, method_name: str) -> str:
        """获取方法的实际类名"""
        info = self.get_method_info(method_name)
        return info.get("class_name", method_name)
    
    def get_task_class_name(self, task_name: str) -> str:
        """获取任务的实际类名"""
        info = self.get_task_info(task_name)
        return info.get("class_name", task_name)
    
    def format_methods_for_display(self) -> str:
        """格式化方法列表用于显示"""
        self._ensure_loaded()
        lines = ["### 可用的算法设计方法\n"]
        for name, info in sorted(self._methods.items()):
            lines.append(f"- **{name}**: {info.get('description', '')}")
        return "\n".join(lines)
    
    def format_tasks_for_display(self) -> str:
        """格式化任务列表用于显示"""
        self._ensure_loaded()
        
        # 按类别分组
        categories = {
            "optimization": "🎯 优化问题",
            "machine_learning": "🤖 机器学习",
            "science_discovery": "🔬 科学发现",
            "other": "📦 其他"
        }
        
        lines = ["### 可用的任务\n"]
        
        for cat_key, cat_name in categories.items():
            tasks_in_cat = [(name, info) for name, info in self._tasks.items() 
                           if info.get("category", "other") == cat_key]
            if tasks_in_cat:
                lines.append(f"\n**{cat_name}**")
                for name, info in sorted(tasks_in_cat):
                    lines.append(f"- `{name}`: {info.get('description', '')}")
        
        return "\n".join(lines)
    
    def search_tasks(self, keyword: str) -> List[str]:
        """搜索任务"""
        self._ensure_loaded()
        keyword = keyword.lower()
        results = []
        for name, info in self._tasks.items():
            if (keyword in name.lower() or 
                keyword in info.get("description", "").lower() or
                keyword in info.get("full_name", "").lower()):
                results.append(name)
        return results
    
    def search_methods(self, keyword: str) -> List[str]:
        """搜索方法"""
        self._ensure_loaded()
        keyword = keyword.lower()
        results = []
        for name, info in self._methods.items():
            if (keyword in name.lower() or 
                keyword in info.get("description", "").lower()):
                results.append(name)
        return results


# 测试代码
if __name__ == "__main__":
    cm = ConfigManager()
    
    print("=== 可用方法 ===")
    for method in cm.get_available_methods():
        info = cm.get_method_info(method)
        print(f"  {method}: {info.get('description', '')}")
    
    print(f"\n=== 可用任务 ({len(cm.get_available_tasks())} 个) ===")
    for cat in cm.get_all_categories():
        tasks = cm.get_tasks_by_category(cat)
        print(f"\n{cat} ({len(tasks)} 个):")
        for task in tasks[:5]:
            info = cm.get_task_info(task)
            print(f"  - {task}: {info.get('description', '')}")
        if len(tasks) > 5:
            print(f"  ... 还有 {len(tasks) - 5} 个")
