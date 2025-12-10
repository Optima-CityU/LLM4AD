"""
LLM4AD Chat Interface - UI Components
可复用的 UI 组件
"""

import streamlit as st
from typing import Dict, Any, List, Optional


def render_config_panel(config_data: Dict[str, Any]):
    """渲染配置面板"""
    st.markdown("""
    <div class="config-panel">
        <h4>📋 参数配置</h4>
        <p>请在右侧面板中配置参数，或告诉我您的需求。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示当前配置状态
    if config_data.get("current_method"):
        st.success(f"✅ 已选择方法: {config_data['current_method']}")
    else:
        st.warning("⚠️ 请选择一个方法")
    
    if config_data.get("current_task"):
        st.success(f"✅ 已选择任务: {config_data['current_task']}")
    else:
        st.warning("⚠️ 请选择一个任务")


def render_result_card(result: Dict[str, Any]):
    """渲染结果卡片"""
    best_score = result.get('best_score', 'N/A')
    total_samples = result.get('total_samples', 'N/A')
    total_time = result.get('total_time', 'N/A')
    best_code = result.get('best_code', '')
    
    # 使用 Streamlit 组件渲染结果
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏆 最佳得分",
            value=f"{best_score:.4f}" if isinstance(best_score, (int, float)) else str(best_score)
        )
    
    with col2:
        st.metric(
            label="📊 总采样数",
            value=total_samples
        )
    
    with col3:
        st.metric(
            label="⏱️ 总耗时",
            value=f"{total_time}秒" if total_time != 'N/A' else 'N/A'
        )
    
    # 显示最佳代码
    if best_code:
        st.markdown("**🔬 最佳算法代码：**")
        st.code(best_code, language="python")


def render_method_card(method_info: Dict[str, Any]):
    """渲染方法信息卡片"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        margin-bottom: 1rem;
    ">
        <h4 style="margin: 0 0 0.5rem 0;">{method_info.get('full_name', method_info.get('name', 'Unknown'))}</h4>
        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">{method_info.get('description', '')}</p>
    </div>
    """, unsafe_allow_html=True)


def render_task_card(task_info: Dict[str, Any]):
    """渲染任务信息卡片"""
    category_colors = {
        "optimization": "#4caf50",
        "machine_learning": "#2196f3",
        "science_discovery": "#ff9800"
    }
    category = task_info.get('category', 'optimization')
    color = category_colors.get(category, "#666")
    
    st.markdown(f"""
    <div style="
        background-color: white;
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="
                background-color: {color}20;
                color: {color};
                padding: 0.2rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.75rem;
            ">{category}</span>
        </div>
        <h4 style="margin: 0 0 0.5rem 0; color: #333;">{task_info.get('full_name', task_info.get('name', 'Unknown'))}</h4>
        <p style="margin: 0; color: #666; font-size: 0.9rem;">{task_info.get('description', '')}</p>
    </div>
    """, unsafe_allow_html=True)


def render_iteration_log(log_data: Dict[str, Any]):
    """渲染迭代日志"""
    iteration = log_data.get('iteration', '?')
    score = log_data.get('score', 'N/A')
    best_score = log_data.get('best_score', 'N/A')
    code = log_data.get('code', '')
    algorithm = log_data.get('algorithm', '')
    
    # 判断是否是当前最佳
    is_best = score is not None and score == best_score
    
    border_color = "#4caf50" if is_best else "#e0e0e0"
    bg_color = "#f1f8e9" if is_best else "#fafafa"
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border: 1px solid {border_color};
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 600;">📊 迭代 #{iteration}</span>
            {'<span style="background: #4caf50; color: white; padding: 0.1rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">🏆 新最佳</span>' if is_best else ''}
        </div>
        <div style="display: flex; gap: 1rem; font-size: 0.85rem; color: #666;">
            <span>得分: <strong>{score if score is not None else 'N/A'}</strong></span>
            <span>最佳: <strong>{best_score}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if code and is_best:
        with st.expander("查看代码", expanded=False):
            st.code(code, language="python")


def render_progress_indicator(message: str, show_spinner: bool = True):
    """渲染进度指示器"""
    if show_spinner:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            background-color: #fff3e0;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        ">
            <div style="
                width: 20px;
                height: 20px;
                border: 2px solid #ff9800;
                border-top-color: transparent;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            "></div>
            <span style="color: #e65100;">{message}</span>
        </div>
        <style>
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.info(message)


def render_methods_list(methods: List[str], config_manager):
    """渲染方法列表"""
    st.markdown("### 可用的方法")
    
    for method_name in methods:
        method_info = config_manager.get_method_info(method_name)
        if method_info:
            render_method_card(method_info)


def render_tasks_list(tasks: List[str], config_manager, category_filter: Optional[str] = None):
    """渲染任务列表"""
    st.markdown("### 可用的任务")
    
    categories = ["optimization", "machine_learning", "science_discovery"]
    category_names = {
        "optimization": "🔧 优化问题",
        "machine_learning": "🤖 机器学习",
        "science_discovery": "🔬 科学发现"
    }
    
    for category in categories:
        if category_filter and category != category_filter:
            continue
            
        category_tasks = [t for t in tasks 
                        if config_manager.get_task_info(t).get('category') == category]
        
        if category_tasks:
            st.markdown(f"#### {category_names.get(category, category)}")
            for task_name in category_tasks:
                task_info = config_manager.get_task_info(task_name)
                if task_info:
                    render_task_card(task_info)


def render_parameter_form(method_name: str, config_manager, current_values: Dict[str, Any] = None):
    """渲染参数配置表单"""
    params = config_manager.get_method_parameters(method_name)
    if not params:
        st.info("该方法没有可配置的参数")
        return {}
    
    current_values = current_values or {}
    values = {}
    
    st.markdown(f"#### {method_name} 参数配置")
    
    for param_name, param_info in params.items():
        default_value = current_values.get(param_name, param_info.get('default'))
        label = param_info.get('label', param_name)
        help_text = param_info.get('help', '')
        
        if param_info['type'] == 'int':
            values[param_name] = st.slider(
                label,
                min_value=param_info.get('min', 1),
                max_value=param_info.get('max', 1000),
                value=default_value,
                help=help_text,
                key=f"param_{method_name}_{param_name}"
            )
        elif param_info['type'] == 'float':
            values[param_name] = st.slider(
                label,
                min_value=float(param_info.get('min', 0)),
                max_value=float(param_info.get('max', 1)),
                value=float(default_value),
                help=help_text,
                key=f"param_{method_name}_{param_name}"
            )
        elif param_info['type'] == 'bool':
            values[param_name] = st.checkbox(
                label,
                value=default_value,
                help=help_text,
                key=f"param_{method_name}_{param_name}"
            )
        elif param_info['type'] == 'select':
            options = param_info.get('options', [])
            values[param_name] = st.selectbox(
                label,
                options=options,
                index=options.index(default_value) if default_value in options else 0,
                help=help_text,
                key=f"param_{method_name}_{param_name}"
            )
    
    return values


def render_chat_bubble(message: str, is_user: bool = False, avatar: str = None):
    """渲染聊天气泡"""
    align = "flex-end" if is_user else "flex-start"
    bg_color = "#e3f2fd" if is_user else "#f5f5f5"
    border_color = "#1976d2" if is_user else "#9e9e9e"
    
    default_avatar = "👤" if is_user else "🤖"
    avatar = avatar or default_avatar
    
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: {align};
        margin: 0.5rem 0;
    ">
        <div style="
            background-color: {bg_color};
            border-left: 3px solid {border_color};
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            max-width: 80%;
        ">
            <div style="
                font-size: 0.75rem;
                color: #666;
                margin-bottom: 0.25rem;
            ">{avatar} {'您' if is_user else 'LLM4AD 助手'}</div>
            <div style="color: #333;">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_code_block(code: str, language: str = "python", title: str = None):
    """渲染代码块"""
    if title:
        st.markdown(f"**{title}**")
    st.code(code, language=language)


def render_status_badge(status: str, text: str = None):
    """渲染状态徽章"""
    colors = {
        "running": ("#fff3e0", "#e65100"),
        "completed": ("#e8f5e9", "#2e7d32"),
        "error": ("#ffebee", "#c62828"),
        "pending": ("#e3f2fd", "#1565c0")
    }
    
    bg_color, text_color = colors.get(status, ("#f5f5f5", "#666"))
    display_text = text or status.capitalize()
    
    st.markdown(f"""
    <span style="
        display: inline-block;
        background-color: {bg_color};
        color: {text_color};
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
    ">{display_text}</span>
    """, unsafe_allow_html=True)


def create_download_button(content: str, filename: str, label: str = "下载"):
    """创建下载按钮"""
    st.download_button(
        label=label,
        data=content,
        file_name=filename,
        mime="text/plain"
    )
