"""
LLM4AD Config Manager - 配置管理器
自动从 llm4ad/method 和 llm4ad/task 中抓取可用的方法和任务
并从对应的 paras.yaml 文件中读取参数配置
"""

import os
import sys
import inspect
import glob
from typing import Dict, List, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yaml
except ImportError:
    yaml = None


class ConfigManager:
    """配置管理器，自动扫描 LLM4AD 的方法和任务"""
    
    # 方法的额外描述信息
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
    
    # 任务分类名称映射
    CATEGORY_NAMES = {
        "optimization": "优化问题",
        "machine_learning": "机器学习/强化学习",
        "science_discovery": "科学发现",
    }
    
    # 任务名称美化映射
    TASK_DISPLAY_NAMES = {
        "OBPEvaluation": "在线装箱问题 (Online Bin Packing)",
        "OBP_2O_Evaluation": "在线装箱问题 (2-Objective)",
        "TSPEvaluation": "旅行商问题 (TSP Construct)",
        "TSPEvaluationCB": "旅行商问题 (CO-Bench)",
        "TSP_GLS_2O_Evaluation": "TSP引导局部搜索 (2-Objective)",
        "CVRPEvaluation": "有容量车辆路径问题 (CVRP)",
        "BP1DEvaluation": "一维装箱问题",
        "BP1DEvaluationCB": "一维装箱问题 (CO-Bench)",
        "BP2DEvaluation": "二维装箱问题",
        "JSSPEvaluation": "作业车间调度问题",
        "CFLPEvaluation": "设施选址问题",
        "AcrobotEvaluation": "Acrobot 控制 (强化学习)",
        "CarMountainEvaluation": "小车爬山 (强化学习)",
        "CarMountainCEvaluation": "小车爬山连续版",
        "PendulumEvaluation": "倒立摆控制",
        "MoonLanderEvaluation": "月球着陆器",
        "FeynmanEvaluation": "费曼符号回归",
        "BactGrowEvaluation": "细菌生长模型",
        "ODE1DEvaluation": "一维常微分方程",
        "Oscillator1Evaluation": "振荡器模型1",
        "Oscillator2Evaluation": "振荡器模型2",
        "StressStrainEvaluation": "应力应变关系",
    }
    
    # 参数类型推断和范围
    PARAM_METADATA = {
        "max_sample_nums": {"type": "int", "min": 5, "max": 10000, "label": "最大采样数", "help": "搜索的最大样本数量"},
        "max_generations": {"type": "int", "min": 1, "max": 100, "label": "最大代数", "help": "演化的最大轮数"},
        "pop_size": {"type": "int", "min": 2, "max": 100, "label": "种群大小", "help": "每代保留的个体数量"},
        "num_samplers": {"type": "int", "min": 1, "max": 16, "label": "采样器数量", "help": "并行采样的线程数"},
        "num_evaluators": {"type": "int", "min": 1, "max": 16, "label": "评估器数量", "help": "并行评估的线程数"},
        "timeout_seconds": {"type": "int", "min": 5, "max": 600, "label": "超时时间(秒)", "help": "单次评估的最大时间"},
        "max_steps": {"type": "int", "min": 100, "max": 10000, "label": "最大步数", "help": "最大仿真步数"},
        "cluster_num": {"type": "int", "min": 1, "max": 20, "label": "簇数量", "help": "聚类的簇数量"},
        "sample_per_cluster": {"type": "int", "min": 1, "max": 20, "label": "每簇采样数", "help": "每个簇的采样数量"},
    }
    
    def __init__(self):
        """初始化配置管理器"""
        self._methods: Dict[str, Dict] = {}
        self._tasks: Dict[str, Dict] = {}
        self._llm4ad_path: str = None
        self._loaded = False
    
    def _get_llm4ad_path(self) -> str:
        """获取 llm4ad 包的路径"""
        if self._llm4ad_path:
            return self._llm4ad_path
        
        try:
            import llm4ad
            self._llm4ad_path = os.path.dirname(llm4ad.__file__)
        except ImportError:
            # 尝试从相对路径找
            self._llm4ad_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'llm4ad'
            )
        return self._llm4ad_path
    
    def _load_yaml(self, yaml_path: str) -> Dict:
        """加载 YAML 文件"""
        if not yaml or not os.path.exists(yaml_path):
            return {}
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"加载 YAML 失败 {yaml_path}: {e}")
            return {}
    
    def _build_param_info(self, param_name: str, default_value: Any) -> Dict:
        """构建参数信息"""
        # 从元数据获取
        if param_name in self.PARAM_METADATA:
            info = dict(self.PARAM_METADATA[param_name])
            info["default"] = default_value
            return info
        
        # 推断类型
        if isinstance(default_value, bool):
            return {"type": "bool", "default": default_value, "label": param_name, "help": ""}
        elif isinstance(default_value, int):
            return {"type": "int", "default": default_value, "min": 1, "max": 10000, "label": param_name, "help": ""}
        elif isinstance(default_value, float):
            return {"type": "float", "default": default_value, "min": 0.0, "max": 1000.0, "label": param_name, "help": ""}
        else:
            return {"type": "str", "default": str(default_value), "label": param_name, "help": ""}
    
    def _ensure_loaded(self):
        """确保已加载方法和任务"""
        if not self._loaded:
            self._scan_methods()
            self._scan_tasks()
            self._loaded = True
    
    def _scan_methods(self):
        """扫描所有方法及其参数"""
        llm4ad_path = self._get_llm4ad_path()
        method_path = os.path.join(llm4ad_path, 'method')
        
        if not os.path.exists(method_path):
            print(f"方法目录不存在: {method_path}")
            return
        
        # 扫描每个方法子目录
        for method_dir in os.listdir(method_path):
            full_path = os.path.join(method_path, method_dir)
            if not os.path.isdir(full_path) or method_dir.startswith('_'):
                continue
            
            # 读取 paras.yaml
            yaml_path = os.path.join(full_path, 'paras.yaml')
            yaml_data = self._load_yaml(yaml_path)
            
            if not yaml_data:
                continue
            
            method_name = yaml_data.get('name', method_dir)
            
            # 构建参数字典
            parameters = {}
            for key, value in yaml_data.items():
                if key == 'name':
                    continue
                # 处理百分号格式（如 timeout_seconds: 20%）
                if isinstance(value, str) and value.endswith('%'):
                    try:
                        value = int(value[:-1])
                    except:
                        pass
                parameters[key] = self._build_param_info(key, value)
            
            # 添加通用参数（如果YAML中没有）
            for common_param in ['num_samplers', 'num_evaluators']:
                if common_param not in parameters:
                    parameters[common_param] = self._build_param_info(common_param, 2)
            
            self._methods[method_name] = {
                "name": method_name,
                "full_name": method_name,
                "description": self.METHOD_DESCRIPTIONS.get(method_name, f"{method_name} 算法设计方法"),
                "class_name": method_name,
                "dir_name": method_dir,
                "parameters": parameters
            }
        
        # 按名称排序
        self._methods = dict(sorted(self._methods.items()))
    
    def _scan_tasks(self):
        """扫描所有任务及其参数"""
        llm4ad_path = self._get_llm4ad_path()
        task_path = os.path.join(llm4ad_path, 'task')
        
        if not os.path.exists(task_path):
            print(f"任务目录不存在: {task_path}")
            return
        
        # 扫描三个类别目录
        categories = ['optimization', 'machine_learning', 'science_discovery']
        
        for category in categories:
            category_path = os.path.join(task_path, category)
            if not os.path.exists(category_path):
                continue
            
            # 扫描每个任务子目录
            for task_dir in os.listdir(category_path):
                full_path = os.path.join(category_path, task_dir)
                if not os.path.isdir(full_path) or task_dir.startswith('_'):
                    continue
                
                # 读取 paras.yaml
                yaml_path = os.path.join(full_path, 'paras.yaml')
                yaml_data = self._load_yaml(yaml_path)
                
                if not yaml_data:
                    continue
                
                class_name = yaml_data.get('name', task_dir)
                
                # 构建参数字典
                parameters = {}
                for key, value in yaml_data.items():
                    if key == 'name':
                        continue
                    # 处理百分号格式
                    if isinstance(value, str) and value.endswith('%'):
                        try:
                            value = int(value[:-1])
                        except:
                            pass
                    parameters[key] = self._build_param_info(key, value)
                
                # 获取显示名称
                display_name = self.TASK_DISPLAY_NAMES.get(class_name, class_name)
                
                self._tasks[class_name] = {
                    "name": class_name,
                    "full_name": display_name,
                    "description": display_name,
                    "category": category,
                    "category_name": self.CATEGORY_NAMES.get(category, category),
                    "class_name": class_name,
                    "dir_name": task_dir,
                    "parameters": parameters
                }
        
        # 按类别和名称排序
        self._tasks = dict(sorted(self._tasks.items(), 
                                   key=lambda x: (x[1]['category'], x[0])))
    
    # ============ 公开 API ============
    
    def get_available_methods(self) -> List[str]:
        """获取所有可用的方法名称"""
        self._ensure_loaded()
        return list(self._methods.keys())
    
    def get_available_tasks(self) -> List[str]:
        """获取所有可用的任务名称"""
        self._ensure_loaded()
        return list(self._tasks.keys())
    
    def get_method_info(self, method_name: str) -> Dict:
        """获取方法的详细信息"""
        self._ensure_loaded()
        return self._methods.get(method_name, {})
    
    def get_task_info(self, task_name: str) -> Dict:
        """获取任务的详细信息"""
        self._ensure_loaded()
        return self._tasks.get(task_name, {})
    
    def get_method_parameters(self, method_name: str) -> Dict[str, Dict]:
        """获取方法的参数配置"""
        self._ensure_loaded()
        return self._methods.get(method_name, {}).get("parameters", {})
    
    def get_task_parameters(self, task_name: str) -> Dict[str, Dict]:
        """获取任务的参数配置"""
        self._ensure_loaded()
        return self._tasks.get(task_name, {}).get("parameters", {})
    
    def get_method_description(self, method_name: str) -> str:
        """获取方法的描述"""
        self._ensure_loaded()
        info = self._methods.get(method_name, {})
        return info.get("description", method_name)
    
    def get_task_description(self, task_name: str) -> str:
        """获取任务的描述"""
        self._ensure_loaded()
        info = self._tasks.get(task_name, {})
        return info.get("description", task_name)
    
    def get_tasks_by_category(self) -> Dict[str, List[str]]:
        """按类别获取任务"""
        self._ensure_loaded()
        result = {}
        for task_name, task_info in self._tasks.items():
            category = task_info.get("category", "other")
            if category not in result:
                result[category] = []
            result[category].append(task_name)
        return result
    
    def get_all_methods_info(self) -> Dict[str, Dict]:
        """获取所有方法的信息（用于Tool显示）"""
        self._ensure_loaded()
        return self._methods
    
    def get_all_tasks_info(self) -> Dict[str, Dict]:
        """获取所有任务的信息（用于Tool显示）"""
        self._ensure_loaded()
        return self._tasks


# 测试代码
if __name__ == "__main__":
    cm = ConfigManager()
    
    print("=" * 60)
    print("可用方法:")
    print("=" * 60)
    for method in cm.get_available_methods():
        info = cm.get_method_info(method)
        params = list(info.get('parameters', {}).keys())
        print(f"  {method}: {info.get('description', '')}")
        print(f"    参数: {params}")
    
    print("\n" + "=" * 60)
    print("可用任务 (按类别):")
    print("=" * 60)
    
    tasks_by_cat = cm.get_tasks_by_category()
    for category, tasks in tasks_by_cat.items():
        cat_name = cm.CATEGORY_NAMES.get(category, category)
        print(f"\n【{cat_name}】")
        for task in tasks:
            info = cm.get_task_info(task)
            params = list(info.get('parameters', {}).keys())
            print(f"  - {task}: {info.get('full_name', '')}")
            if params:
                print(f"      参数: {params}")
