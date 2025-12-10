"""
LLM4AD Algorithm Runner - 算法设计运行器
封装 LLM4AD 的运行逻辑，支持流式输出设计过程
"""

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime
from typing import Dict, Any, Generator, Optional
from io import StringIO

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz


class OutputCapture:
    """捕获标准输出用于流式显示"""
    
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue
        self._original_stdout = sys.stdout
        self._buffer = StringIO()
    
    def write(self, text):
        self._original_stdout.write(text)
        self._buffer.write(text)
        # 解析输出并放入队列
        if text.strip():
            self.output_queue.put({
                "type": "output",
                "text": text
            })
    
    def flush(self):
        self._original_stdout.flush()
        self._buffer.flush()
    
    def get_captured(self):
        return self._buffer.getvalue()


class StreamingProfiler:
    """流式输出的 Profiler，用于捕获算法设计过程"""
    
    def __init__(self, output_queue: queue.Queue, base_profiler=None):
        self.output_queue = output_queue
        self.base_profiler = base_profiler
        self._num_samples = 0
        self._best_score = float('-inf')
        self._best_function = None
        self._best_program = None
        self._start_time = time.time()
        self._last_update_time = time.time()
    
    def register_function(self, function, program: str = '', **kwargs):
        """注册一个评估过的函数 - 这是核心的流式输出点"""
        self._num_samples += 1
        score = function.score
        
        # 检查是否是新的最佳
        is_new_best = False
        if score is not None and score > self._best_score:
            self._best_score = score
            self._best_function = function
            self._best_program = program
            is_new_best = True
        
        # 计算耗时
        current_time = time.time()
        elapsed = current_time - self._start_time
        iter_time = current_time - self._last_update_time
        self._last_update_time = current_time
        
        # 获取代码内容和算法描述
        code_str = None
        algorithm_desc = None
        docstring = None
        if function is not None:
            try:
                code_str = str(function)
            except:
                code_str = None
            # 获取算法描述
            try:
                algorithm_desc = getattr(function, 'algorithm', None)
                docstring = getattr(function, 'docstring', None)
            except:
                pass
        
        # 发送详细的更新到队列
        self.output_queue.put({
            "type": "iteration",
            "iteration": self._num_samples,
            "score": score,
            "best_score": self._best_score if self._best_score != float('-inf') else None,
            "is_new_best": is_new_best,
            "code": code_str if is_new_best else None,
            "algorithm": algorithm_desc if is_new_best else None,
            "docstring": docstring if is_new_best else None,
            "elapsed_time": round(elapsed, 1),
            "iter_time": round(iter_time, 2),
        })
        
        # 如果有基础 profiler，也调用它
        if self.base_profiler:
            self.base_profiler.register_function(function, program, **kwargs)
    
    def record_parameters(self, llm, prob, method):
        """记录参数"""
        self.output_queue.put({
            "type": "info",
            "message": f"参数已记录: LLM={llm.__class__.__name__}, Method={method.__class__.__name__}"
        })
        if self.base_profiler:
            self.base_profiler.record_parameters(llm, prob, method)
    
    def finish(self):
        """完成运行"""
        total_time = time.time() - self._start_time
        
        # 获取最佳算法的描述
        best_algorithm = None
        best_docstring = None
        if self._best_function:
            try:
                best_algorithm = getattr(self._best_function, 'algorithm', None)
                best_docstring = getattr(self._best_function, 'docstring', None)
            except:
                pass
        
        self.output_queue.put({
            "type": "finished",
            "best_score": self._best_score if self._best_score != float('-inf') else None,
            "best_code": str(self._best_function) if self._best_function else None,
            "best_algorithm": best_algorithm,
            "best_docstring": best_docstring,
            "best_program": self._best_program,
            "total_samples": self._num_samples,
            "total_time": round(total_time, 2)
        })
        if self.base_profiler:
            self.base_profiler.finish()
    
    def get_logger(self):
        if self.base_profiler:
            return self.base_profiler.get_logger()
        return None


class AlgorithmRunner:
    """算法设计运行器"""
    
    def __init__(self, 
                 method_name: str,
                 task_name: str,
                 llm_config: Dict[str, Any],
                 parameters: Dict[str, Any] = None):
        """
        初始化算法运行器
        
        Args:
            method_name: 方法名称（如 EoH, FunSearch 等）
            task_name: 任务名称（如 OBPEvaluation 等）
            llm_config: LLM 配置（host, key, model）
            parameters: 方法参数
        """
        self.method_name = method_name
        self.task_name = task_name
        self.llm_config = llm_config
        self.parameters = parameters or {}
        
        self._output_queue = queue.Queue()
        self._is_running = False
        self._runner_thread = None
        self._result = None
    
    def _import_components(self):
        """动态导入 LLM4AD 组件"""
        import inspect
        
        # 清理可能导致库冲突的环境变量
        import os as _os
        env_keys_to_clean = ['DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH']
        original_env = {}
        for key in env_keys_to_clean:
            if key in _os.environ:
                original_env[key] = _os.environ.pop(key)
        
        try:
            import llm4ad
            
            from llm4ad.task import import_all_evaluation_classes
            from llm4ad.method import import_all_method_classes_from_subfolders
            from llm4ad.tools.llm import import_all_llm_classes_from_subfolders
            from llm4ad.tools.profiler.profile import ProfilerBase
            
            # 获取 llm4ad 包的路径
            llm4ad_path = _os.path.dirname(llm4ad.__file__)
            
            # 导入所有类
            import_all_evaluation_classes(_os.path.join(llm4ad_path, 'task'))
            import_all_method_classes_from_subfolders(_os.path.join(llm4ad_path, 'method'))
            import_all_llm_classes_from_subfolders(_os.path.join(llm4ad_path, 'tools/llm'))
            
            # 获取所有可用类
            components = {}
            for module in [llm4ad.tools.llm, llm4ad.tools.profiler, llm4ad.task, llm4ad.method]:
                components.update({name: obj for name, obj in vars(module).items() if inspect.isclass(obj)})
            
            return components
        finally:
            # 恢复环境变量
            for key, val in original_env.items():
                _os.environ[key] = val
    
    def _run_internal(self):
        """内部运行方法（在线程中执行）"""
        try:
            # 导入组件
            components = self._import_components()
            
            # 获取类 - 直接使用类名（现在 config_manager 已确保类名正确）
            method_class = components.get(self.method_name)
            eval_class = components.get(self.task_name)
            llm_class = components.get('HttpsApi')
            
            if not method_class:
                self._output_queue.put({
                    "type": "error",
                    "message": f"未找到方法类: {self.method_name}。可用方法: {[k for k in components.keys() if not k.startswith('_')][:10]}"
                })
                return
            
            if not eval_class:
                self._output_queue.put({
                    "type": "error", 
                    "message": f"未找到任务类: {self.task_name}"
                })
                return
            
            # 创建 LLM 实例
            llm_instance = llm_class(
                host=self.llm_config.get('host', 'api.bltcy.top'),
                key=self.llm_config.get('key', ''),
                model=self.llm_config.get('model', 'gpt-4o-mini')
            )
            
            # 创建评估器实例
            eval_instance = eval_class()
            
            # 创建流式 Profiler
            streaming_profiler = StreamingProfiler(self._output_queue)
            
            # 发送开始信号
            self._output_queue.put({
                "type": "progress",
                "value": 0,
                "message": "初始化算法设计环境..."
            })
            
            # 准备方法参数
            method_params = {
                'llm': llm_instance,
                'evaluation': eval_instance,
                'profiler': streaming_profiler,
            }
            
            # 添加用户指定的参数
            for key, value in self.parameters.items():
                if key not in ['llm', 'evaluation', 'profiler']:
                    method_params[key] = value
            
            # 设置默认参数（如果未指定）
            if 'max_sample_nums' not in method_params:
                method_params['max_sample_nums'] = 50
            if 'num_samplers' not in method_params:
                method_params['num_samplers'] = 2
            if 'num_evaluators' not in method_params:
                method_params['num_evaluators'] = 2
            
            # 创建方法实例
            method_instance = method_class(**method_params)
            
            # 发送启动信号
            self._output_queue.put({
                "type": "started",
                "method": self.method_name,
                "task": self.task_name,
                "message": f"🚀 开始使用 {self.method_name} 设计 {self.task_name} 的算法..."
            })
            
            # 运行
            method_instance.run()
            
            # 完成 - 调用 profiler.finish()
            streaming_profiler.finish()
            
        except Exception as e:
            import traceback
            self._output_queue.put({
                "type": "error",
                "message": f"运行出错: {str(e)}\n{traceback.format_exc()}"
            })
        finally:
            self._is_running = False
            self._output_queue.put({"type": "done"})
    
    def run_with_stream(self) -> Generator[Dict[str, Any], None, None]:
        """运行算法并流式返回输出"""
        self._is_running = True
        
        # 在后台线程中运行
        self._runner_thread = threading.Thread(target=self._run_internal, daemon=True)
        self._runner_thread.start()
        
        # 进度估计变量
        last_iteration = 0
        max_samples = self.parameters.get('max_sample_nums', 100)
        
        # 从队列中读取输出
        while self._is_running or not self._output_queue.empty():
            try:
                update = self._output_queue.get(timeout=0.1)
                
                if update["type"] == "done":
                    break
                
                # 直接转发所有类型的更新
                yield update
                    
            except queue.Empty:
                continue
        
        # 等待线程完成
        if self._runner_thread and self._runner_thread.is_alive():
            self._runner_thread.join(timeout=5)
    
    def run_sync(self) -> Dict[str, Any]:
        """同步运行算法"""
        results = list(self.run_with_stream())
        
        # 查找最终结果
        for r in reversed(results):
            if r.get("type") == "result":
                return r.get("data", {})
        
        return {"error": "未能获取结果"}
    
    def stop(self):
        """停止运行"""
        self._is_running = False


class MockAlgorithmRunner:
    """模拟算法运行器，用于测试和演示"""
    
    def __init__(self, 
                 method_name: str,
                 task_name: str,
                 llm_config: Dict[str, Any],
                 parameters: Dict[str, Any] = None):
        self.method_name = method_name
        self.task_name = task_name
        self.llm_config = llm_config
        self.parameters = parameters or {}
    
    def run_with_stream(self) -> Generator[Dict[str, Any], None, None]:
        """模拟流式输出"""
        max_iterations = self.parameters.get('max_sample_nums', 20)
        
        yield {
            "type": "progress",
            "value": 0,
            "message": "初始化算法设计环境..."
        }
        time.sleep(0.5)
        
        yield {
            "type": "progress",
            "value": 5,
            "message": "开始算法设计..."
        }
        
        best_score = float('-inf')
        mock_codes = [
            '''def priority(item, bins):
    """优先级函数 v1"""
    return bins - item''',
            '''def priority(item, bins):
    """优先级函数 v2"""
    return (bins - item) / (bins + 1)''',
            '''def priority(item, bins):
    """优先级函数 v3"""
    mask = bins >= item
    scores = np.where(mask, bins - item, -np.inf)
    return scores''',
            '''def priority(item, bins):
    """优先级函数 v4 - 改进版"""
    remaining = bins - item
    mask = remaining >= 0
    # 偏好剩余空间小的箱子
    scores = np.where(mask, -remaining / (bins + 1), -np.inf)
    return scores''',
        ]
        
        for i in range(min(max_iterations, 10)):
            time.sleep(0.3)  # 模拟计算时间
            
            # 模拟得分提升
            import random
            score = -120 + i * 5 + random.random() * 10
            if score > best_score:
                best_score = score
            
            progress = 10 + int(85 * (i + 1) / max_iterations)
            
            yield {
                "type": "progress",
                "value": progress,
                "message": f"第 {i+1} 次迭代，当前最佳得分: {best_score:.2f}"
            }
            
            yield {
                "type": "log",
                "iteration": i + 1,
                "score": score,
                "best_score": best_score,
                "code": mock_codes[i % len(mock_codes)],
                "algorithm": f"启发式策略 v{i+1}"
            }
        
        yield {
            "type": "progress",
            "value": 100,
            "message": "算法设计完成！"
        }
        
        # 最终结果
        yield {
            "type": "result",
            "data": {
                "best_score": best_score,
                "best_code": mock_codes[-1],
                "total_samples": max_iterations,
                "total_time": max_iterations * 0.3
            }
        }


def create_runner(method_name: str,
                  task_name: str, 
                  llm_config: Dict[str, Any],
                  parameters: Dict[str, Any] = None,
                  use_mock: bool = False) -> AlgorithmRunner:
    """创建算法运行器的工厂函数"""
    if use_mock:
        return MockAlgorithmRunner(method_name, task_name, llm_config, parameters)
    return AlgorithmRunner(method_name, task_name, llm_config, parameters)
