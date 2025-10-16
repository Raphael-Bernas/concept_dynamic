# On the Learning Representation of LLM: A Concept-Oriented Study

## About This Project

This repository contains the work conducted during my end-of-term internship as part of my double diploma program at **MVA (Mathematics, Vision, and Learning) Master** at ENS Paris-Saclay and **ENSTA Paris**. The internship was carried out at the **MICS laboratory** under the supervision of **Céline Hudelot** and **Fanny Jourdan** (IRT Toulouse).

### Project Title
**"On the learning representation of LLM: A concept-oriented study"**

### Abstract
Explainable Artificial Intelligence (XAI) has emerged from the need for transparent and interpretable AI predictions, especially as models grow in complexity. This need is particularly present in the context of Deep Learning (DL), where model decisions are often opaque. Mechanistic interpretability, a subfield of XAI, addresses this challenge by seeking to understand the internal structure and functioning of Large Language Models (LLMs) to explain their outputs. However, most existing approaches focus on post-hoc analysis, applied after training, and relatively few have explored the use of mechanistic methods to monitor the learning process itself. In this work, we investigate the evolution of model internals during training using the EuroBERT model and its intermediate checkpoints. By applying mechanistic interpretability techniques throughout the training phase, we aim to derive insights into the learning dynamics of Large Language Models (LLMs), ultimately contributing to the development of more structured learning frameworks and learning dynamics.

**Keywords:** Learning, LLM, Concept, Explainability, Mechanistic Interpretability, AI

## Acknowledgments

I would like to express my sincere gratitude to:

- The **Interpreto team** ([GitHub repository](https://github.com/FOR-sight-ai/interpreto)) for their help and for providing essential tools and frameworks that formed the foundation of this research.
- The **EuroBERT team** ([arXiv paper](https://arxiv.org/abs/2503.05500)) for their help and for providing the model checkpoints that enabled this study.

Their contributions were instrumental in making this project possible, providing the necessary building blocks for our concept-oriented analysis of LLM learning dynamics.

## Repository Structure

This repository is organized into three main folders, each serving a specific purpose in the research workflow:

### 📁 `bash/`
Contains bash scripts designed to facilitate the execution of different experimental methods described in the internship report. These scripts provide an easy-to-use interface for running various analyses and experiments without having to manually configure parameters each time.

Note that to access the methods from the report you need to use bash files as follows :

- **Concepts extraction** : `concept_dynamic/bash/main_decomposition.sh` with `decomposition_method="concepts_retrieval"`.
- **Space Representation Encoding** : `concept_dynamic/bash/main_higher_decomposition.sh` with `decomposition_method="codes_retrieval"`.
- **Concept Activation Vector (CAV)** : `concept_dynamic/bash/main_cav_decomposition.sh` or `concept_dynamic/bash/main_higher_decomposition.sh` with `decomposition_method="cav_retrieval"`.
- **Concept Equivalence** : `concept_dynamic/bash/main_higher_decomposition.sh` with `decomposition_method="concepts_retrieval"`.
- **With weighted correlation** : If you want to have access to weighted correlation, you need `concept_dynamic/bash/main_decomposition.sh` with `decomposition_method="mixed_retrieval"`.

### 📁 `code/`
Houses the core implementation files used throughout the experiments. The main components include:

- **`main.py`**: Central execution script that coordinates the overall experimental pipeline
- **`eurobert_XAI.py`**: Specialized module for EuroBERT model analysis and explainability methods
- **`cav_utilis.py`**: Utilities for Concept Activation Vector (CAV) analysis and manipulation
- **`process.py`**: Processing functions for model activations concepts already retrieved
- **`plot.py`**: Visualization utilities for generating figures and analysis plots
- **`utils.py`**: General utility functions used across different modules
- **`manifold/`**: Subdirectory containing specialized manifold analysis tools:
  - **`main.py`**: Main execution script for manifold-based experiments
  - **`manifold.py`**: Core manifold learning and dimensionality reduction implementations
  - **`json_handle.py`**: JSON file handling utilities for data serialization and loading

### 📁 `other/`
Contains supplementary materials to facilitate project setup and accessibility:

- **`requirements_conda_cross_platform.yml`**: Conda environment specification for cross-platform compatibility
- **`requirements_pip.txt`**: pip requirements file for easy package installation
- **`EuroBERT_XAI_EASY.ipynb`**: A user-friendly Jupyter notebook designed for Google Colab execution, providing an accessible entry point to the project's main functionalities

## Getting Started

To reproduce the experiments or explore the methodologies presented in this work, please refer to the bash scripts in the `bash/` folder or use the Google Colab notebook provided in the `other/` folder for a quick start.

