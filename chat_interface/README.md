# LLM4AD Chat Interface

基于对话的自动算法设计交互界面。通过自然语言对话的方式，帮助用户使用 LLM4AD 框架进行自动算法设计。

## 🌟 特性

- **🗣️ 对话式交互**：通过自然语言与 AI 助手对话，选择方法、配置任务和参数
- **🔄 流式输出**：实时展示算法设计过程，包括每次迭代的得分和生成的代码
- **🎨 科研风格界面**：简洁专业的 UI 设计，适合科研场景
- **🎭 演示模式**：无需 API Key 即可体验完整功能
- **⚙️ 灵活配置**：支持对话配置和手动配置两种方式

## 🚀 快速开始

### 1. 安装依赖

```bash
cd LLM4AD
pip install streamlit pytz
# 或者
pip install -r chat_interface/requirements.txt
```

### 2. 运行应用

```bash
cd LLM4AD
streamlit run chat_interface/app.py
```

### 3. 访问界面

浏览器会自动打开，或手动访问 `http://localhost:8501`

## 📖 使用指南

### 对话方式

直接在聊天框中输入您的需求：

- "我想用 EoH 方法解决在线装箱问题"
- "有哪些可用的方法？"
- "有哪些优化任务？"
- "开始运行"

### 手动配置

1. 点击侧边栏的"配置"按钮
2. 在右侧面板中选择方法和任务
3. 调整参数
4. 点击"开始运行"

### 配置 LLM API

在侧边栏中配置：

- **API Host**: 如 `api.bltcy.top` 或 `api.openai.com`
- **API Key**: 您的 API 密钥
- **模型**: 选择要使用的模型

## 🛠️ 可用方法

| 方法 | 描述 |
|------|------|
| EoH | 启发式演化，通过交叉变异操作进化算法 |
| FunSearch | Google DeepMind 的函数搜索方法 |
| HillClimb | 爬山算法，简单有效的局部搜索 |
| RandSample | 随机采样，作为基准方法 |
| MEoH | 多专家启发式演化 |
| MoEaD | 多目标演化算法 |
| NSGA2 | 非支配排序遗传算法 |

## 📋 可用任务

### 优化问题
- **OBPEvaluation**: 在线装箱问题
- **TSPConstruct**: 旅行商问题
- **CVRPConstruct**: 有容量车辆路径问题
- **KnapsackConstruct**: 背包问题
- **BP1DConstruct**: 一维装箱问题
- **BP2DConstruct**: 二维装箱问题

### 机器学习
- **CarMountain**: 山地车控制
- **Acrobot**: 双摆控制
- **Pendulum**: 倒立摆控制

### 科学发现
- **FeynmanSRSD**: 符号回归

## 🏗️ 项目结构

```
chat_interface/
├── __init__.py          # 包初始化
├── app.py               # 主应用入口（简化版）
├── chat_app.py          # 主应用入口（完整版）
├── chat_agent.py        # 对话 Agent，负责意图理解
├── config_manager.py    # 配置管理器
├── algorithm_runner.py  # 算法运行器，支持流式输出
├── components.py        # UI 组件
├── requirements.txt     # 依赖列表
└── README.md            # 说明文档
```

## 🔧 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│  (chat_app.py / app.py)                                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌───────────────┐    ┌──────────────────────────────┐  │
│  │  ChatAgent    │    │     ConfigManager            │  │
│  │  (外层 LLM)   │    │     (方法/任务/参数管理)     │  │
│  └───────┬───────┘    └──────────────────────────────┘  │
│          │                                               │
│          ▼                                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │              AlgorithmRunner                       │  │
│  │              (流式输出封装)                        │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                               │
├──────────────────────────┼──────────────────────────────┤
│                          ▼                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │                   LLM4AD Core                      │  │
│  │  (EoH, FunSearch, Evaluations, LLM APIs...)       │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                               │
│                          ▼                               │
│                    内层 LLM 调用                          │
└─────────────────────────────────────────────────────────┘
```

## 📝 开发说明

### 添加新方法

在 `config_manager.py` 的 `METHODS` 字典中添加：

```python
"NewMethod": {
    "name": "NewMethod",
    "full_name": "New Method Name",
    "description": "方法描述",
    "class_name": "NewMethod",
    "parameters": {
        "param1": {"type": "int", "default": 10, ...}
    }
}
```

### 添加新任务

在 `config_manager.py` 的 `TASKS` 字典中添加：

```python
"NewTask": {
    "name": "NewTask",
    "full_name": "New Task Name",
    "description": "任务描述",
    "category": "optimization",
    "class_name": "NewTaskEvaluation",
    "module_path": "llm4ad.task.optimization.new_task.evaluation",
    "parameters": {...}
}
```

## 📄 License

MIT License - 详见项目根目录 LICENSE 文件

## 🙏 致谢

- [LLM4AD](https://github.com/Optima-CityU/llm4ad) - 自动算法设计框架
- [Streamlit](https://streamlit.io/) - Web 应用框架
