"""
LLM4AD Chat Interface - Main Application
一个基于对话的自动算法设计交互界面

使用方法:
    cd LLM4AD
    streamlit run chat_interface/chat_app.py
"""

import os
import sys
import time
import streamlit as st
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_interface.chat_agent import ChatAgent
from chat_interface.config_manager import ConfigManager
from chat_interface.components import render_config_panel, render_result_card
from chat_interface.algorithm_runner import AlgorithmRunner

# 页面配置
st.set_page_config(
    page_title="LLM4AD - 自动算法设计助手",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义样式
def load_custom_css():
    css = """
    <style>
    /* 整体风格 - 科研风格，简洁专业 */
    .main {
        background-color: #fafbfc;
    }
    
    /* 聊天消息样式 */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: #e8f4fd;
        border-left: 4px solid #1976d2;
    }
    
    .chat-message.assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #424242;
    }
    
    .chat-message .message-content {
        margin-top: 0.5rem;
    }
    
    /* 代码块样式 */
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
        margin: 0.5rem 0;
    }
    
    /* 结果卡片样式 */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .result-card h3 {
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .result-card .score {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* 进度指示器 */
    .progress-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        font-size: 0.9rem;
    }
    
    .progress-indicator .dot {
        width: 8px;
        height: 8px;
        background-color: #4caf50;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* 配置面板样式 */
    .config-panel {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    
    /* 状态徽章 */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-badge.running {
        background-color: #fff3e0;
        color: #e65100;
    }
    
    .status-badge.completed {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    
    .status-badge.error {
        background-color: #ffebee;
        color: #c62828;
    }
    
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* 标题样式 */
    .main-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    /* 算法代码展示 */
    .algorithm-display {
        background-color: #282c34;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .algorithm-display pre {
        color: #abb2bf;
        margin: 0;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* 流式输出样式 */
    .stream-output {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #333;
    }
    
    /* 迭代进度 */
    .iteration-info {
        background-color: #f0f7ff;
        border: 1px solid #cce5ff;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def init_session_state():
    """初始化 session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = None
    
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    if "algorithm_runner" not in st.session_state:
        st.session_state.algorithm_runner = None
    
    if "current_config" not in st.session_state:
        st.session_state.current_config = {
            "method": None,
            "task": None,
            "llm": None,
            "parameters": {}
        }
    
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    
    if "show_config_panel" not in st.session_state:
        st.session_state.show_config_panel = False
    
    if "run_results" not in st.session_state:
        st.session_state.run_results = []


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("### ⚙️ LLM 配置")
        
        # 外层 LLM 配置（用于对话）
        st.markdown("#### 对话 LLM")
        outer_host = st.text_input("API Host", value="api.bltcy.top", key="outer_host")
        outer_key = st.text_input("API Key", type="password", key="outer_key")
        outer_model = st.selectbox(
            "模型", 
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-sonnet"],
            key="outer_model"
        )
        
        st.markdown("---")
        
        # 内层 LLM 配置（用于算法设计）
        st.markdown("#### 算法设计 LLM")
        use_same_llm = st.checkbox("使用相同配置", value=True)
        
        if use_same_llm:
            inner_host = outer_host
            inner_key = outer_key
            inner_model = outer_model
        else:
            inner_host = st.text_input("API Host", value="api.bltcy.top", key="inner_host")
            inner_key = st.text_input("API Key", type="password", key="inner_key")
            inner_model = st.selectbox(
                "模型",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                key="inner_model"
            )
        
        # 存储配置
        st.session_state.current_config["llm"] = {
            "outer": {"host": outer_host, "key": outer_key, "model": outer_model},
            "inner": {"host": inner_host, "key": inner_key, "model": inner_model}
        }
        
        st.markdown("---")
        
        # 当前状态显示
        st.markdown("### 📊 当前状态")
        config = st.session_state.current_config
        
        method_status = f"✅ {config['method']}" if config['method'] else "❌ 未选择"
        task_status = f"✅ {config['task']}" if config['task'] else "❌ 未选择"
        
        st.markdown(f"**方法:** {method_status}")
        st.markdown(f"**任务:** {task_status}")
        
        if st.session_state.is_running:
            st.markdown('<span class="status-badge running">🔄 运行中</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 快捷操作
        st.markdown("### 🚀 快捷操作")
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_config = {
                "method": None,
                "task": None,
                "llm": st.session_state.current_config.get("llm"),
                "parameters": {}
            }
            st.rerun()
        
        if st.button("📝 手动配置", use_container_width=True):
            st.session_state.show_config_panel = not st.session_state.show_config_panel
            st.rerun()


def render_chat_message(message):
    """渲染聊天消息"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # 检查是否有特殊内容类型
            if isinstance(content, dict):
                if content.get("type") == "algorithm_result":
                    render_result_card(content.get("data", {}))
                elif content.get("type") == "config_form":
                    render_config_panel(content.get("data", {}))
                else:
                    st.markdown(content.get("text", ""))
            else:
                st.markdown(content)


def get_welcome_message():
    """获取欢迎消息"""
    return """
👋 **欢迎使用 LLM4AD 自动算法设计平台！**

我是您的AI助手，可以帮助您：
- 🎯 **选择算法设计方法**：如 EoH（启发式演化）、FunSearch、随机采样等
- 📋 **配置优化任务**：如在线装箱问题、TSP路径规划、车辆路径问题等
- ⚙️ **设置参数**：采样数量、种群大小、迭代次数等
- 🔬 **运行算法设计**：实时展示设计过程和结果

---

**开始方式：**
1. 💬 直接告诉我您想解决什么问题，我会为您推荐合适的方法
2. 🔧 或者说"显示配置面板"来手动设置参数
3. ❓ 如果有疑问，可以问我"有哪些可用的方法？"或"有哪些任务？"

**示例对话：**
- "我想用EoH方法解决在线装箱问题"
- "帮我设计一个TSP问题的启发式算法"
- "列出所有可用的优化方法"

请在下方输入您的需求开始吧！ 👇
"""


def main():
    """主函数"""
    load_custom_css()
    init_session_state()
    
    # 初始化 ChatAgent
    if st.session_state.chat_agent is None:
        llm_config = st.session_state.current_config.get("llm", {}).get("outer", {})
        if llm_config.get("key"):
            st.session_state.chat_agent = ChatAgent(
                host=llm_config.get("host", "api.bltcy.top"),
                api_key=llm_config.get("key", ""),
                model=llm_config.get("model", "gpt-4o-mini")
            )
    
    # 渲染侧边栏
    render_sidebar()
    
    # 主内容区
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 标题
        st.markdown('<h1 class="main-title">🧬 LLM4AD 自动算法设计助手</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">通过对话的方式，让AI帮您设计高效的算法</p>', unsafe_allow_html=True)
        
        # 显示欢迎消息（如果是第一次）
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(get_welcome_message())
        
        # 显示历史消息
        for message in st.session_state.messages:
            render_chat_message(message)
        
        # 如果正在运行，显示进度
        if st.session_state.is_running:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
                <div class="progress-indicator">
                    <div class="dot"></div>
                    <span>正在设计算法中，请稍候...</span>
                </div>
                """, unsafe_allow_html=True)
        
        # 聊天输入
        if prompt := st.chat_input("输入您的需求...", disabled=st.session_state.is_running):
            # 检查 API Key
            llm_config = st.session_state.current_config.get("llm", {}).get("outer", {})
            if not llm_config.get("key"):
                st.error("⚠️ 请先在侧边栏配置 API Key")
            else:
                # 添加用户消息
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # 如果 agent 未初始化，现在初始化
                if st.session_state.chat_agent is None:
                    st.session_state.chat_agent = ChatAgent(
                        host=llm_config.get("host", "api.bltcy.top"),
                        api_key=llm_config.get("key", ""),
                        model=llm_config.get("model", "gpt-4o-mini")
                    )
                
                # 更新 agent 配置
                st.session_state.chat_agent.update_config(
                    host=llm_config.get("host"),
                    api_key=llm_config.get("key"),
                    model=llm_config.get("model")
                )
                
                # 获取响应
                with st.spinner("思考中..."):
                    response = st.session_state.chat_agent.chat(
                        prompt,
                        st.session_state.current_config,
                        st.session_state.config_manager
                    )
                
                # 处理响应
                if response.get("action") == "run_algorithm":
                    # 需要运行算法
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("message", "好的，我将开始为您设计算法...")
                    })
                    st.session_state.is_running = True
                    st.session_state.current_config.update(response.get("config", {}))
                    st.rerun()
                    
                elif response.get("action") == "show_config":
                    # 显示配置面板
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {
                            "type": "config_form",
                            "data": response.get("config_data", {})
                        }
                    })
                    st.session_state.show_config_panel = True
                    st.rerun()
                    
                elif response.get("action") == "update_config":
                    # 更新配置
                    st.session_state.current_config.update(response.get("config", {}))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("message", "配置已更新。")
                    })
                    st.rerun()
                    
                else:
                    # 普通对话响应
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("message", "我不太理解您的意思，请再说一遍。")
                    })
                    st.rerun()
    
    with col2:
        # 手动配置面板
        if st.session_state.show_config_panel:
            st.markdown("### 📋 参数配置")
            
            with st.form("config_form"):
                # 方法选择
                methods = st.session_state.config_manager.get_available_methods()
                selected_method = st.selectbox(
                    "选择方法",
                    options=[""] + methods,
                    index=0 if not st.session_state.current_config.get("method") 
                          else methods.index(st.session_state.current_config["method"]) + 1 
                          if st.session_state.current_config.get("method") in methods else 0
                )
                
                # 任务选择
                tasks = st.session_state.config_manager.get_available_tasks()
                selected_task = st.selectbox(
                    "选择任务",
                    options=[""] + tasks,
                    index=0 if not st.session_state.current_config.get("task")
                          else tasks.index(st.session_state.current_config["task"]) + 1
                          if st.session_state.current_config.get("task") in tasks else 0
                )
                
                st.markdown("---")
                st.markdown("**方法参数**")
                
                # 根据选择的方法显示参数
                if selected_method:
                    method_params = st.session_state.config_manager.get_method_parameters(selected_method)
                    param_values = {}
                    for param_name, param_info in method_params.items():
                        if param_info["type"] == "int":
                            param_values[param_name] = st.number_input(
                                param_info.get("label", param_name),
                                value=param_info.get("default", 10),
                                min_value=param_info.get("min", 1),
                                max_value=param_info.get("max", 1000),
                                help=param_info.get("help", "")
                            )
                        elif param_info["type"] == "bool":
                            param_values[param_name] = st.checkbox(
                                param_info.get("label", param_name),
                                value=param_info.get("default", True),
                                help=param_info.get("help", "")
                            )
                
                submitted = st.form_submit_button("💾 保存配置", use_container_width=True)
                
                if submitted:
                    if selected_method:
                        st.session_state.current_config["method"] = selected_method
                    if selected_task:
                        st.session_state.current_config["task"] = selected_task
                    if selected_method:
                        st.session_state.current_config["parameters"] = param_values
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ 配置已保存！\n- 方法: {selected_method or '未选择'}\n- 任务: {selected_task or '未选择'}"
                    })
                    st.rerun()
            
            # 运行按钮
            if st.session_state.current_config.get("method") and st.session_state.current_config.get("task"):
                if st.button("🚀 开始设计算法", use_container_width=True, type="primary"):
                    st.session_state.is_running = True
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"🚀 开始使用 **{st.session_state.current_config['method']}** 方法设计 **{st.session_state.current_config['task']}** 任务的算法..."
                    })
                    st.rerun()
    
    # 如果正在运行，执行算法设计
    if st.session_state.is_running:
        run_algorithm_design()


def run_algorithm_design():
    """运行算法设计"""
    config = st.session_state.current_config
    llm_config = config.get("llm", {}).get("inner", {})
    
    if not llm_config.get("key"):
        st.error("⚠️ 请先配置 API Key")
        st.session_state.is_running = False
        return
    
    # 创建算法运行器
    runner = AlgorithmRunner(
        method_name=config.get("method"),
        task_name=config.get("task"),
        llm_config=llm_config,
        parameters=config.get("parameters", {})
    )
    
    # 创建输出容器
    output_container = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 运行算法并流式输出
        result = None
        for update in runner.run_with_stream():
            if update["type"] == "progress":
                progress_bar.progress(update["value"])
                status_text.text(update.get("message", ""))
            elif update["type"] == "log":
                with output_container.container():
                    st.markdown(f"""
                    <div class="iteration-info">
                        <strong>📊 第 {update.get('iteration', '?')} 次迭代</strong><br>
                        当前得分: {update.get('score', 'N/A')}<br>
                        最佳得分: {update.get('best_score', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if update.get("code"):
                        st.code(update["code"], language="python")
            elif update["type"] == "result":
                result = update["data"]
        
        # 完成
        st.session_state.is_running = False
        progress_bar.progress(100)
        status_text.text("✅ 算法设计完成！")
        
        if result:
            # 添加结果消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "type": "algorithm_result",
                    "data": result
                }
            })
            
            # 添加文字总结
            summary = f"""
🎉 **算法设计完成！**

**最终结果：**
- 🏆 最佳得分: **{result.get('best_score', 'N/A')}**
- 📊 总采样数: {result.get('total_samples', 'N/A')}
- ⏱️ 总耗时: {result.get('total_time', 'N/A')}秒

**最佳算法代码：**
```python
{result.get('best_code', '# 无法获取代码')}
```

您可以继续与我对话，优化参数或尝试其他方法。
            """
            st.session_state.messages.append({
                "role": "assistant",
                "content": summary
            })
        
        st.rerun()
        
    except Exception as e:
        st.session_state.is_running = False
        st.error(f"❌ 运行出错: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ 算法设计过程中出现错误: {str(e)}\n\n请检查配置后重试。"
        })
        st.rerun()


if __name__ == "__main__":
    main()
