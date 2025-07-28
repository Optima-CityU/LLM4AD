import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os


class ImageSwitcher:
    def __init__(self, root):
        self.root = root
        self.root.title("折线图浏览器")
        self.root.geometry("800x600")

        # 创建图表列表
        self.figures = []
        self.create_figures()

        # 当前显示的图表索引
        self.current_index = 0

        # 创建界面
        self.create_widgets()
        self.show_current_figure()

    def create_figures(self):
        """生成5张示例折线图"""
        for i in range(5):
            fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

            # 生成随机数据
            x = np.linspace(0, 10, 100)
            y = np.sin(x) + np.random.normal(0, 0.2, 100) * (i + 1) / 2

            # 绘制折线图
            ax.plot(x, y, 'o-', markersize=4, label=f"数据集 {i + 1}")
            ax.set_title(f"折线图示例 {i + 1}")
            ax.set_xlabel("X轴")
            ax.set_ylabel("Y轴")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            self.figures.append(fig)
            plt.close(fig)  # 关闭图形以避免显示在非Tkinter窗口中

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 标题标签
        self.title_label = ttk.Label(main_frame, text="", font=("Arial", 14))
        self.title_label.pack(pady=10)

        # 图片显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧按钮
        btn_prev = ttk.Button(image_frame, text="←", width=5,
                              command=self.show_previous)
        btn_prev.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 图表显示区域
        self.canvas_frame = ttk.Frame(image_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右侧按钮
        btn_next = ttk.Button(image_frame, text="→", width=5,
                              command=self.show_next)
        btn_next.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # 状态栏
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    def show_current_figure(self):
        """显示当前图表"""
        # 清除旧图表
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # 创建新图表
        fig = self.figures[self.current_index]
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 更新标题
        self.title_label.config(text=f"折线图 {self.current_index + 1}/{len(self.figures)}")

    def show_previous(self):
        """显示上一张图表"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_figure()

    def show_next(self):
        """显示下一张图表"""
        if self.current_index < len(self.figures) - 1:
            self.current_index += 1
            self.show_current_figure()



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSwitcher(root)
    root.mainloop()