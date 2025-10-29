<div align="center">
<h1 align="center">
<img src="./assets/figs/logo.png" alt="LLM4AD Logo" style="width: 90%; height: auto;">
</h1>
<h1 align="center">
Eoh-Java
</h1>

[![Releases](https://img.shields.io/badge/Release-v1.0-blue)](https://github.com/Optima-CityU/LLM4AD/releases)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Optima-CityU/LLM4AD/pulls)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
[![License](https://img.shields.io/badge/License-MIT-important)](https://github.com/Optima-CityU/LLM4AD/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/llm4ad-doc/badge/?version=latest)](https://llm4ad-doc.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Optima-CityU/llm4ad/blob/main/example/online_bin_packing/online_bin_packing_tutorial.ipynb)

[Website](http://www.llm4ad.com/)
| [Documentation](https://llm4ad-doc.readthedocs.io/en/latest/)
| [Examples](https://github.com/Optima-CityU/LLM4AD/tree/main/example)
| [GUI](https://github.com/Optima-CityU/LLM4AD/tree/main/GUI)

</div>
<br>

## 📖 Introduction 

Eoh-Java 旨在于 LLM4AD 平台中扩展演进启发式 (Evolution of Heuristics, EoH) 方法，使其能够分析、编译并改进 Java 项目。

本 README 将通过一个具体案例——优化 Java 版本的 AILS-II 算法——来演示如何使用大语言模型（LLM）辅助进行 Java 算法的设计与迭代。

AILS-II 的相关任务文件位于 llm4ad/task/optimization/ails2 目录下。如果您需要自定义新的优化任务，请参考此目录结构来设计您自己的 evaluation.py 和 template.py。

## 🎁 Requirements & Installation

### Python 环境

我们提供了 `environment.yaml` 文件，您可以使用 Conda 快速配置 Python 环境。
请在 LLM4AD 项目根目录下启动终端，并运行：
```bash
conda env create -f environment.yaml
````

### Java 环境 (JDK)

需要 **Java Development Kit (JDK)** 才能编译和评估 Java 代码。

1.  **安装 JDK**：
    如果您尚未安装 Java 环境，请从官方渠道下载并安装 JDK (例如 [Oracle JDK](https://www.oracle.com/java/technologies/downloads/) 或 [OpenJDK](https://openjdk.org/install/))。推荐使用 JDK 8 或更高版本。

2.  **验证安装**：
    安装完成后，打开命令行工具，分别输入 `java -version` 和 `javac -version`。如果均能正确显示版本号，则表示安装成功。

3.  **获取 `javac` 目录路径 (重要)**：
    您需要找到 `javac` 编译器所在的**目录**的绝对路径，并在后续步骤中（例如配置 `evaluation.py` 时）使用它。

      * **Windows**: 运行 `where javac`。如果输出为 `C:\Program Files\...\javapath\javac.exe`，则您需要记录的路径是 `C:\Program Files\...\javapath`。
      * **Linux/macOS**: 运行 `which javac`。如果输出为 `/usr/bin/javac`，则您需要记录的路径是 `/usr/bin`。

## 💻 Example Usage

### Quick Start:

> [!Note]
> Configure your LLM api before running the script. For example:
>
> 1) Set `host`: 'api.deepseek.com'
> 2) Set `key`: 'your api key'
> 3) Set `model`: 'deepseek-chat'

```python
import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.ails2 import Ails2Evaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh_java import EoH_Java
from llm4ad.method.eoh_java import EoH_java_Profiler

def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='xxxx',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=60)

    task = Ails2Evaluation()

    method = EoH_Java(llm=llm,
                      profiler=EoH_java_Profiler(log_dir='logs', log_style='complex'),
                      evaluation=task,
                      max_sample_nums=48,
                      max_generations=5,
                      pop_size=8,
                      num_samplers=2,
                      num_evaluators=8,
                      debug_mode=True)
    method.run()
```

## 🚀 如何改进您自己的 Java 项目

如果您希望使用 Eoh-Java 改进您自己的 Java 项目，请遵循以下步骤：

### 1\. 创建任务 (Task)

1.  在 `llm4ad/task` 目录下创建一个新的任务目录 (例如：`llm4ad/task/my_java_project`)。
2.  将您**完整且可运行**的 Java 项目复制到这个新目录中 (例如：将 `CVRPLIB_2025_AILSII` 项目放入 `my_java_project` 目录)。

### 2\. 实现 `template.py`

在您的新任务目录中创建 `template.py` 文件，并设置以下关键变量：

  * `java_dir`: 需要改进的 Java 项目的**根目录名称**。
    > 例如：`java_dir = "CVRPLIB_2025_AILSII"`
  * `aim_java_relative_path`: 您希望 LLM **优化**的 Java 文件对与Java项目根目录的**相对路径**。
    > 例如，如果目标文件是 `CVRPLIB_2025_AILSII/Method/AILS-II/src/DiversityControl/DistAdjustment.java`，则设置为：
    > `aim_java_relative_path = os.path.join('Method', 'AILS-II','src', 'DiversityControl', 'DistAdjustment.java')`
  * `task_description`: 详细描述目标 Java 脚本的**原始功能**，需要LLM做的任务，以及您**期望的改进方向**（可选）。
  * `template_program`: 提供需要改进的**原始 Java 脚本内容**。这可以是完整的 Java 文件，也可以是您希望 LLM 遵循的代码框架。这将作为 LLM 初始化的参考。

### 3\. 实现 `evaluation.py`

在您的新任务目录中创建 `evaluation.py` 文件。这是 Eoh-Java 用来**评估**新生成代码性能的核心。

1.  **参考示例**:
      * 如果您现在是在优化 AILS-II 的某个模块，请仔细检查 `llm4ad/task/optimization/ails2/evaluation.py` 文件中的每一个 `# TODO` 标记，并根据您的实际情况进行修改（例如 `javac` 路径）。
2.  **实现 `evaluate()` 功能**:
      * 如果您使用的是新项目，请参考 AILS-II 的 `evaluation.py` 来实现您自己的 `evaluate()` 函数。
      * 此函数必须能够：
        1.  接收修改后的 Java 代码。
        2.  编译和运行该 Java 代码。
        3.  返回一个**性能评分 (fitness score)**。**评分越高，表示性能越好。**


## 🚀 如果你需要修改AILSII的其他模块而不是其他算法

请跳过上述的第一步，直接进行第二和第三步。

## 🪪 Licence

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details. Parts of this project use code licensed under the Apache License 2.0.

## ✨ Reference

If you find LLM4AD helpful please cite:

```bibtex
@article{liu2024llm4ad,
    title = {LLM4AD: A Platform for Algorithm Design with Large Language Model},
    author = {Fei Liu and Rui Zhang and Zhuoliang Xie and Rui Sun and Kai Li and Xi Lin and Zhenkun Wang and Zhichao Lu and Qingfu Zhang},
    year = {2024},
    eprint = {2412.17287},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url = {https://arxiv.org/abs/2412.17287},
}
```


