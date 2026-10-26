# SPDX-License-Identifier: BSD-2-Clause
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Licensed under BSD-2-Clause; see the project LICENSE file.
# Academic citation is appreciated, but is not a license condition.
# Suggested reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For voluntary commercial collaboration inquiries, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from setuptools import setup, find_packages
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
VERSION = runpy.run_path(str(ROOT / 'llm4ad' / '_version.py'))['__version__']

CORE_REQUIRES = [
    'numpy>=1.24,<3',
    'matplotlib',
    'pytz',
    'tqdm',
]

EXTRAS_REQUIRE = {
    'gui': ['ttkbootstrap', 'PyYAML'],
    'numba': ['numba'],
    'tensorboard': ['torch', 'tensorboard'],
    'wandb': ['wandb'],
    'openai': ['openai'],
    'local-vllm': [
        'requests',
        'psutil',
        'torch',
        'flask',
        'flask-cors',
        'transformers',
        'vllm',
    ],
    'local-ollama': ['langchain-ollama'],
    'funsearch': ['scipy'],
    'multi-objective': ['pymoo'],
    'meoh': [
        'pymoo',
        'codebleu',
        'tree-sitter-python==0.23',
    ],
    'partevo': [
        'codebleu',
        'tree-sitter-python==0.23',
        'scikit-learn',
        'seaborn',
        'torch',
        'transformers',
    ],
    'llamea': [
        'llamea==1.2.0',
        'ConfigSpace',
    ],
    'machine-learning': ['gymnasium[box2d]'],
    'science-discovery': ['pandas', 'scipy', 'sympy'],
    'pymoo-task': ['pymoo'],
    'co-bench': [
        'datasets',
        'huggingface_hub',
        'httpx',
        'httpcore',
        'networkx',
        'scipy',
    ],
    'tsp-gls': ['numba', 'scipy'],
    'test': ['coverage>=7,<8', 'build>=1,<2', 'twine>=5,<8'],
    'benchmark': ['psutil>=5'],
    'docs': ['sphinx>=7,<10', 'myst-parser>=2,<6', 'furo'],
}

# Install every optional feature with:
#   pip install ".[all]"
EXTRAS_REQUIRE['all'] = sorted({
    package
    for extra, packages in EXTRAS_REQUIRE.items() if extra not in {'test', 'docs'}
    for package in packages
})

setup(
    name='llm4ad',
    version=VERSION,
    author='LLM4AD Developers',
    description='Large Language Model for Algorithm Design Platform ',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/Optima-CityU/LLM4AD',
    project_urls={
        'Source': 'https://github.com/Optima-CityU/LLM4AD',
        'Documentation': 'https://llm4ad-doc.readthedocs.io/en/latest/',
        'Issues': 'https://github.com/Optima-CityU/LLM4AD/issues',
        'Changelog': 'https://github.com/Optima-CityU/LLM4AD/blob/main/CHANGELOG.md',
    },
    license='BSD-2-Clause AND Apache-2.0',
    license_files=['LICENSE', 'LICENSES/Apache-2.0.txt', 'NOTICE', 'THIRD_PARTY_NOTICES.md'],
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*']),
    package_data={'': ['*.yaml', '*.json']},
    package_dir={'': '.'},
    python_requires='>=3.9,<3.14',
    install_requires=CORE_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
