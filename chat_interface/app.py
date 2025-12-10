"""
LLM4AD Chat Interface - Main Application
一个基于对话的自动算法设计交互界面

所有对话都通过外部LLM实现，使用OpenAI Function Calling风格的工具调用。

使用方法:
    cd LLM4AD
    streamlit run chat_interface/app.py
"""

import os
import sys
import time
import json
import streamlit as st
import pandas as pd
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from chat_interface.tool_agent import ToolCallingAgent
from chat_interface.config_manager import ConfigManager
from chat_interface.algorithm_runner import create_runner

# 页面配置
st.set_page_config(
    page_title="LLM4AD - 自动算法设计助手",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """初始化 session state"""
    defaults = {
        "messages": [],
        "chat_agent": None,
        "config_manager": ConfigManager(),
        "current_config": {
            "method": None,
            "task": None,
            "llm": {"outer": {}, "inner": {}},
            "parameters": {}
        },
        "is_running": False,
        "show_config_panel": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## ⚙️ 配置")
        
        st.markdown("---")
        
        # LLM 配置
        st.markdown("### 🤖 LLM 设置")
        st.markdown("*外部对话 + 内部算法设计都使用此 API*")
        
        api_host = st.text_input(
            "API Host", 
            value=st.session_state.current_config["llm"].get("outer", {}).get("host", "api.bltcy.top"),
            key="api_host"
        )
        
        api_key = st.text_input(
            "API Key", 
            type="password",
            value=st.session_state.current_config["llm"].get("outer", {}).get("key", ""),
            key="api_key",
            help="OpenAI 兼容的 API Key"
        )
        
        api_model = st.selectbox(
            "模型",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "deepseek-chat"],
            key="api_model"
        )
        
        # 更新配置
        st.session_state.current_config["llm"] = {
            "outer": {"host": api_host, "key": api_key, "model": api_model},
            "inner": {"host": api_host, "key": api_key, "model": api_model}
        }
        
        # 如果 API Key 变了，重置 agent
        if api_key and st.session_state.chat_agent:
            if st.session_state.chat_agent.api_key != api_key:
                st.session_state.chat_agent = None
        
        st.markdown("---")
        
        # 当前状态
        st.markdown("### 📊 当前状态")
        config = st.session_state.current_config
        
        method = config.get("method")
        task = config.get("task")
        
        if method:
            st.success(f"✅ 方法: {method}")
        else:
            st.warning("❌ 方法: 未选择")
            
        if task:
            st.success(f"✅ 任务: {task}")
        else:
            st.warning("❌ 任务: 未选择")
        
        if st.session_state.is_running:
            st.info("🔄 运行中...")
        
        st.markdown("---")
        
        # 快捷操作
        st.markdown("### 🚀 快捷操作")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.messages = []
                st.session_state.current_config["method"] = None
                st.session_state.current_config["task"] = None
                st.session_state.current_config["parameters"] = {}
                st.rerun()
        
        with col2:
            if st.button("⚙️ 配置", use_container_width=True):
                st.session_state.show_config_panel = not st.session_state.show_config_panel
                st.rerun()


def get_welcome_message():
    """获取欢迎消息"""
    return """
👋 **欢迎使用 LLM4AD 自动算法设计助手！**

我是一个基于大语言模型的智能助手，可以帮助您：
- 🎯 选择算法设计方法（如 EoH、FunSearch 等）
- 📋 配置优化任务（如装箱问题、TSP 等）
- ⚙️ 设置参数并运行算法设计
- 📊 实时展示设计过程和最终结果

**开始方式（所有对话都由 LLM 处理）：**
- 直接描述您的需求，例如："我想用EoH方法解决在线装箱问题"
- 或询问："有哪些可用的方法？"、"什么任务适合我的场景？"

⚠️ **请先在左侧边栏配置 API Key**，然后开始对话。
"""


def process_user_input(user_input: str):
    """处理用户输入 - 通过 LLM 和工具调用实现"""
    config = st.session_state.current_config
    
    # 获取 API 配置
    api_key = config.get("llm", {}).get("outer", {}).get("key", "")
    
    # 如果有 API Key，使用 ToolCallingAgent
    if api_key:
        # 确保 agent 已初始化
        if st.session_state.chat_agent is None:
            llm_config = config.get("llm", {}).get("outer", {})
            st.session_state.chat_agent = ToolCallingAgent(
                host=llm_config.get("host", "api.bltcy.top"),
                api_key=api_key,
                model=llm_config.get("model", "gpt-4o-mini")
            )
        
        # 通过 agent 处理
        agent = st.session_state.chat_agent
        result = agent.chat(user_input, config, st.session_state.config_manager)
        
        # 同步配置变更
        st.session_state.current_config = agent.current_config
        
        return result
    
    # 没有 API Key 时的简单回退逻辑
    return {
        "action": "chat",
        "message": "⚠️ 请先在侧边栏配置 API Key，然后所有对话将通过 LLM 处理。"
    }


def run_algorithm_design():
    """运行算法设计并流式输出 - 精美版"""
    config = st.session_state.current_config
    
    # 获取 LLM 配置 - 内外层统一使用侧边栏配置
    llm_config = config.get("llm", {}).get("outer", {})
    if not llm_config.get("key"):
        llm_config = {
            "host": st.session_state.get("api_host", "api.bltcy.top"),
            "key": st.session_state.get("api_key", ""),
            "model": st.session_state.get("api_model", "gpt-4o-mini")
        }
    
    # 创建运行器
    runner = create_runner(
        method_name=config.get("method"),
        task_name=config.get("task"),
        llm_config=llm_config,
        parameters=config.get("parameters", {}),
        use_mock=False
    )
    
    # ========== 界面布局 ==========
    # 顶部状态
    st.markdown(f"### 🚀 正在运行: **{config.get('method')}** → **{config.get('task')}**")
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # 实时统计卡片
    stats_cols = st.columns(4)
    stat_iteration = stats_cols[0].empty()
    stat_best = stats_cols[1].empty()
    stat_time = stats_cols[2].empty()
    stat_rate = stats_cols[3].empty()
    
    st.markdown("---")
    
    # 左右布局：日志 + 曲线
    col_log, col_chart = st.columns([1, 1])
    
    with col_log:
        st.markdown("### 📋 迭代日志")
        log_placeholder = st.empty()
    
    with col_chart:
        st.markdown("### 📈 收敛曲线")
        chart_placeholder = st.empty()
    
    st.markdown("---")
    
    # 最佳代码展示区
    st.markdown("### 🔥 当前最佳算法")
    best_info_placeholder = st.empty()
    best_code_placeholder = st.empty()
    
    # ========== 数据存储 ==========
    best_result = None
    logs = []
    iteration_logs = []
    score_history = []  # 用于绘制收敛曲线
    best_history = []   # 最佳得分历史
    current_best_code = None
    current_best_desc = None
    success_count = 0
    
    try:
        for update in runner.run_with_stream():
            update_type = update.get("type")
            
            if update_type == "info":
                status_container.info(f"ℹ️ {update.get('message', '')}")
            
            elif update_type == "started":
                status_container.success(f"🚀 {update.get('message', '算法设计已启动')}")
                progress_bar.progress(5)
            
            elif update_type == "iteration":
                iteration = update.get("iteration", 0)
                score = update.get("score")
                best_score = update.get("best_score")
                is_new_best = update.get("is_new_best", False)
                code = update.get("code", "")
                algorithm_desc = update.get("algorithm", "")
                docstring = update.get("docstring", "")
                elapsed = update.get("elapsed_time", 0)
                iter_time = update.get("iter_time", 0)
                
                # 记录数据
                score_history.append(score if score is not None else None)
                best_history.append(best_score if best_score is not None else (best_history[-1] if best_history else None))
                if score is not None:
                    success_count += 1
                
                # 计算进度
                max_samples = config.get("parameters", {}).get("max_sample_nums", 50)
                progress = min(5 + int(90 * iteration / max_samples), 95)
                progress_bar.progress(progress)
                
                # 更新状态栏
                score_str = f"{score:.4f}" if score is not None else "失败"
                best_str = f"{best_score:.4f}" if best_score is not None else "N/A"
                status_container.info(f"⏳ 迭代 {iteration}/{max_samples} | 当前: {score_str} | 最佳: {best_str}")
                
                # 更新统计卡片
                stat_iteration.metric("🔄 迭代", f"{iteration}/{max_samples}")
                stat_best.metric("🏆 最佳得分", best_str)
                stat_time.metric("⏱️ 耗时", f"{elapsed:.1f}s")
                rate = f"{100*success_count/iteration:.0f}%" if iteration > 0 else "N/A"
                stat_rate.metric("✅ 成功率", rate)
                
                # 构建日志条目
                if is_new_best:
                    log_entry = {"icon": "🏆", "iter": iteration, "score": score_str, 
                                 "status": "✨新最佳", "time": f"{iter_time:.2f}s", 
                                 "is_best": True, "code": code}
                    if code:
                        current_best_code = code
                        current_best_desc = algorithm_desc or docstring
                elif score is not None:
                    log_entry = {"icon": "📊", "iter": iteration, "score": score_str, 
                                 "status": "", "time": f"{iter_time:.2f}s", "is_best": False}
                else:
                    log_entry = {"icon": "❌", "iter": iteration, "score": "失败", 
                                 "status": "错误", "time": f"{iter_time:.2f}s", "is_best": False}
                
                iteration_logs.append(log_entry)
                logs.append(update)
                
                # 更新日志表格（显示最近12条）
                display_logs = iteration_logs[-12:]
                log_md = "| | # | 得分 | 状态 | 耗时 |\n|:---:|:---:|:---:|:---:|:---:|\n"
                for log in display_logs:
                    status_cell = f"**{log['status']}**" if log['status'] else "-"
                    score_cell = f"**{log['score']}**" if log.get('is_best') else log['score']
                    log_md += f"| {log['icon']} | {log['iter']} | {score_cell} | {status_cell} | {log['time']} |\n"
                log_placeholder.markdown(log_md)
                
                # 更新收敛曲线
                if len(score_history) > 1:
                    chart_data = pd.DataFrame({
                        '迭代得分': score_history,
                        '最佳得分': best_history
                    })
                    chart_placeholder.line_chart(chart_data, use_container_width=True)
                
                # 更新最佳代码展示
                if current_best_code:
                    if current_best_desc:
                        best_info_placeholder.success(f"**算法描述**: {current_best_desc}")
                    with best_code_placeholder.expander(f"📝 查看代码 (得分: {best_str})", expanded=False):
                        st.code(current_best_code, language="python")
            
            elif update_type == "finished":
                best_result = update
                progress_bar.progress(100)
                status_container.success("✅ 算法设计完成!")
            
            elif update_type == "error":
                st.error(f"❌ 错误: {update.get('message', '未知错误')}")
                st.session_state.is_running = False
                return
        
        # ========== 最终结果展示 ==========
        progress_bar.progress(100)
        status_container.success("✅ 算法设计完成!")
        
        if best_result:
            st.markdown("---")
            st.markdown("## 🎉 最终结果")
            
            # 最终统计
            final_cols = st.columns(4)
            score = best_result.get('best_score')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            final_cols[0].metric("🏆 最佳得分", score_str)
            final_cols[1].metric("📊 总采样", best_result.get('total_samples', 'N/A'))
            total_time = best_result.get('total_time', 0)
            final_cols[2].metric("⏱️ 总耗时", f"{total_time:.1f}s" if isinstance(total_time, (int, float)) else "N/A")
            rate = f"{100*success_count/len(logs):.0f}%" if logs else "N/A"
            final_cols[3].metric("✅ 成功率", rate)
            
            # 算法描述
            best_algorithm = best_result.get('best_algorithm')
            best_docstring = best_result.get('best_docstring')
            if best_algorithm or best_docstring:
                st.markdown("### 💡 算法描述")
                st.info(best_algorithm or best_docstring)
            
            # 最佳代码
            if best_result.get('best_code'):
                st.markdown("### 🔬 最佳算法代码")
                st.code(best_result['best_code'], language="python")
                
                # 下载按钮
                col_dl1, col_dl2, _ = st.columns([1, 1, 2])
                with col_dl1:
                    st.download_button(
                        label="📥 下载代码",
                        data=best_result['best_code'],
                        file_name=f"best_{config.get('task', 'alg')}.py",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col_dl2:
                    log_json = json.dumps(logs, indent=2, ensure_ascii=False, default=str)
                    st.download_button(
                        label="📋 导出日志",
                        data=log_json,
                        file_name=f"log_{config.get('task', 'alg')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            # 最佳版本历史
            best_logs = [l for l in iteration_logs if l.get('is_best') and l.get('code')]
            if len(best_logs) > 1:
                with st.expander(f"📈 最佳版本演进 ({len(best_logs)} 次突破)", expanded=False):
                    for i, bl in enumerate(best_logs, 1):
                        st.markdown(f"**#{i}** 迭代 {bl['iter']} | 得分: {bl['score']}")
                        st.code(bl['code'], language="python")
                        if i < len(best_logs):
                            st.markdown("---")
            
            # 保存到消息历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🎉 **算法设计完成！**\n\n- 最佳得分: **{score_str}**\n- 总采样: {best_result.get('total_samples', 'N/A')}\n- 耗时: {total_time:.1f}s"
            })
        
    except Exception as e:
        st.error(f"❌ 运行出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        st.session_state.is_running = False


def render_config_panel_ui():
    """渲染配置面板 UI"""
    config = st.session_state.current_config
    config_manager = st.session_state.config_manager
    
    st.markdown("### ⚙️ 手动配置")
    
    # 方法选择
    methods = config_manager.get_available_methods()
    current_method_idx = methods.index(config["method"]) if config["method"] in methods else 0
    
    selected_method = st.selectbox(
        "选择方法",
        options=methods,
        index=current_method_idx if config["method"] else 0,
        key="config_method_select"
    )
    
    # 任务选择
    tasks = config_manager.get_available_tasks()
    current_task_idx = tasks.index(config["task"]) if config["task"] in tasks else 0
    
    selected_task = st.selectbox(
        "选择任务",
        options=tasks,
        index=current_task_idx if config["task"] else 0,
        key="config_task_select"
    )
    
    # 参数设置
    st.markdown("#### 参数设置")
    params = config_manager.get_method_parameters(selected_method)
    param_values = {}
    
    for param_name, param_info in params.items():
        current_val = config.get("parameters", {}).get(param_name, param_info.get("default"))
        
        if param_info["type"] == "int":
            param_values[param_name] = st.slider(
                param_info.get("label", param_name),
                min_value=param_info.get("min", 1),
                max_value=param_info.get("max", 1000),
                value=current_val,
                help=param_info.get("help", ""),
                key=f"config_param_{param_name}"
            )
    
    # 保存按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存配置", use_container_width=True, type="primary"):
            config["method"] = selected_method
            config["task"] = selected_task
            config["parameters"] = param_values
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ 配置已保存！\n- 方法: {selected_method}\n- 任务: {selected_task}"
            })
            st.session_state.show_config_panel = False
            st.rerun()
    
    with col2:
        if st.button("🚀 开始运行", use_container_width=True, 
                    disabled=not (selected_method and selected_task)):
            config["method"] = selected_method
            config["task"] = selected_task
            config["parameters"] = param_values
            st.session_state.is_running = True
            st.session_state.show_config_panel = False
            st.rerun()


def main():
    """主函数"""
    init_session_state()
    
    # 初始化 ToolCallingAgent（如果有 API Key）
    llm_config = st.session_state.current_config.get("llm", {}).get("outer", {})
    if llm_config.get("key") and st.session_state.chat_agent is None:
        st.session_state.chat_agent = ToolCallingAgent(
            host=llm_config.get("host", "api.bltcy.top"),
            api_key=llm_config.get("key", ""),
            model=llm_config.get("model", "gpt-4o-mini")
        )
    
    # 渲染侧边栏
    render_sidebar()
    
    # 主布局
    if st.session_state.show_config_panel:
        col1, col2 = st.columns([2, 1])
    else:
        col1, col2 = st.columns([3, 1])
    
    with col1:
        # 标题
        st.markdown("# 🧬 LLM4AD 自动算法设计助手")
        st.markdown("*通过对话的方式，让AI帮您设计高效的算法*")
        st.markdown("---")
        
        # 如果正在运行，显示算法设计过程
        if st.session_state.is_running:
            run_algorithm_design()
        else:
            # 显示欢迎消息或历史
            if not st.session_state.messages:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(get_welcome_message())
            
            # 显示历史消息
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                    st.markdown(msg["content"])
            
            # 聊天输入
            if prompt := st.chat_input("输入您的需求..."):
                # 添加用户消息
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # 显示用户消息
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                
                # 处理输入
                with st.spinner("思考中..."):
                    response = process_user_input(prompt)
                
                # 处理响应
                action = response.get("action", "chat")
                message = response.get("message", "")
                tool_calls = response.get("tool_calls", "")
                
                # 如果有工具调用，显示
                if tool_calls:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.caption(tool_calls)
                
                if action == "run_algorithm":
                    st.session_state.messages.append({"role": "assistant", "content": message})
                    st.session_state.is_running = True
                    st.rerun()
                else:
                    st.session_state.messages.append({"role": "assistant", "content": message})
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message)
                    st.rerun()
    
    with col2:
        if st.session_state.show_config_panel:
            render_config_panel_ui()


if __name__ == "__main__":
    main()
