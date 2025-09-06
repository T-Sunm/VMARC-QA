# VMARC-QA: A Knowledge-based Multi-agent Approach for Vietnamese VQA with Rationale Explanations

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://github.com/T-Sunm/VMARC-QA/blob/main/2026_AICI_VMARC-QA.pdf)
[![Code](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/T-Sunm/VMARC-QA)
[![Dataset](https://img.shields.io/badge/Dataset-ViVQA--X-blue)](https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## 📖 Introduction

This is the official implementation of the paper **"A knowledge-based multi-agent approach for Vietnamese VQA with rationale explanations"**.

Visual Question Answering with Natural Language Explanations (VQA-NLE) requires a model to predict a correct answer and provide a coherent rationale. This challenge is amplified for the Vietnamese language due to a lack of specialized datasets and methods.

To address this, we introduce **VMARC-QA** (Vietnamese Multi-Agent Rationale-driven Consensus for Question Answering), the first knowledge-based multi-agent framework designed for Vietnamese VQA-NLE. Key features of VMARC-QA include:

-   **Hierarchical Multi-Agent Architecture**: A parallel ensemble of three agents (Junior, Senior, Manager) with different capabilities for evidence gathering.
-   **Evidence-to-Rationale Bottleneck**: A mechanism that forces the model to generate a rationale based solely on collected evidence before predicting a final answer.
-   **Dual-Stream Consensus Mechanism**: A system that aggregates answers via weighted voting while simultaneously validating the semantic consistency of rationales to ensure reliability.

On the **ViVQA-X** benchmark, VMARC-QA demonstrates exceptional performance in answer accuracy, achieving **64.8%**. This result is a significant improvement of over **11 percentage points** compared to strong prior baselines such as LSTM-Generative (53.8%) and NLX-GPT (53.7%). Regarding explanation quality, VMARC-QA remains highly competitive with a BERTScore of **76.0**, nearly matching the 76.3 achieved by NLX-GPT. This indicates that our approach significantly enhances the reasoning capability for correct answer prediction while generating explanations with high semantic fidelity.

The overall architecture of VMARC-QA is shown in the figure below:

![Framework](./assets/Fullpipeline.pdf)
*Figure 1: Overview of the VMARC-QA Framework. Three agents independently generate answer-rationale pairs, which are then aggregated by a dual-stream consensus mechanism to produce the final output.*

## Table of Contents
* [🚀 Quick Start](#-quick-start)
* [⚙️ Installation](#️-installation)
* [📦 Data Preparation](#-data-preparation)
* [▶️ Usage](#️-usage)
* [📊 Evaluation](#-evaluation)
* [📈 Results](#-results)
* [📁 Repository Structure](#-repository-structure)
* [📜 Citation](#-citation)
* [📝 License](#-license)

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone --recurse-submodules https://github.com/T-Sunm/Visual-Multi-Agent-Knowledge-QA-.git
cd Visual-Multi-Agent-Knowledge-QA-

# 2. Install dependencies (main environment)
conda create -n vivqa-minh python=3.10 -y
conda activate vivqa-minh
pip install -r requirements.txt

# 3. Run a sample query
# (Ensure your data is set up and local LLM server is running if needed)
bash scripts/full_system.sh
```

## ⚙️ Installation

This section provides a detailed guide to setting up the environment.

### 1. Prerequisites
- **Conda**: To manage dependencies in an isolated environment.
- **Python 3.10+**
- **API Keys**: For any external services you wish to use (e.g., OpenAI).

### 2. Environment Setup
```bash
# Clone the repository and navigate into the directory
git clone --recurse-submodules https://github.com/T-Sunm/Visual-Multi-Agent-Knowledge-QA-.git
cd Visual-Multi-Agent-Knowledge-QA-
```

**Step A: Main Graph Environment**
```bash
# Create a new conda environment named 'vivqa-minh' with Python 3.10
conda create -n vivqa-minh python=3.10 -y
conda activate vivqa-minh
pip install -r requirements.txt
```

**Step B: VQA Tool Environment**
```bash
# Navigate to the submodule directory
cd ViVQA-X

# Create and set up the environment for the VQA tool
conda create -n mak_vivqax_lstm -y
conda activate mak_vivqax_lstm
pip install -r requirements.txt
pip install fastapi==0.115.12 uvicorn[standard]==0.34.2 python-multipart
```

## 📦 Data Preparation

This system is designed to work with Vietnamese VQA datasets. The integrated **ViVQA-X**, which uses images from the MS COCO dataset.

### Data Aquisition
*(TODO: Add instructions or scripts for downloading the data.)*

### Expected Structure
The scripts expect the data to be organized in a specific structure. The code often references a root directory like `/mnt/VLAI_data/`. A typical structure would be:
```
/path/to/your/data/
├── COCO_Images/
│   ├── train2014/
│   └── val2014/
│
└── ViVQA-X/
    ├── ViVQA-X_train.json
    └── ViVQA-X_val.json
```

## ▶️ Usage

The system can be run with a local or remote LLM.

### Step 1: Run the VQA Tool Server
Activate the `mak_vivqax_lstm` environment and start the API server from the `ViVQA-X` directory.
```bash
conda activate mak_vivqax_lstm
cd api
python main.py
```

### Step 2: Run the LLM Server (for Local Models)
If you are using a local model with VLLM, open a new terminal, activate the `vivqa-minh` conda environment, and start the server. The following is an example command for the Qwen model.

```bash
# Command to serve a local LLM with VLLM
conda activate vivqa-minh

CUDA_VISIBLE_DEVICES=1 \
vllm serve Qwen/Qwen3-1.7B \
  --port 1234 \
  --dtype auto \
  --gpu-memory-utilization 0.45 \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
```

### Step 3: Run Sample Query
Once the environment is set up and servers are running, you can run a query from the command line.

Go to `scripts/full_system.sh`, change `--samples` to the number of samples you want to test. If set to `0`, it will run on the full dataset.
```bash
bash scripts/full_system.sh
```

## 📊 Evaluation
*(TODO: Add instructions on how to run evaluation scripts from the `src/evaluation/` directory to reproduce the results reported in the paper.)*

## 📈 Results

Performance comparison on the **ViVQA-X test set**.  
Metrics include: BLEU (B1–B4), METEOR (M), ROUGE-L (R-L), CIDEr (C), SPICE (S), BERTScore-MaxRef (BS-MR), and Answer Accuracy (Acc).  

| Method              | B1    | B2    | B3    | B4    | M     | R-L   | C     | S    | BS-MR | Acc   |
|---------------------|-------|-------|-------|-------|-------|-------|-------|------|-------|-------|
| Heuristic Baseline  | 8.46  | 3.0   | 1.3   | 0.6   | 8.5   | 7.9   | 0.5   | 0.6  | 70.8  | 10.1  |
| LSTM-Generative     | 22.6  | 11.7  | 6.2   | 3.2   | 16.4  | 23.7  | 34.1  | 4.3  | 72.2  | 53.8  |
| **NLX-GPT**         | **42.4** | **27.8** | **18.5** | **12.4** | 20.4  | **32.8** | **51.4** | **5.0** | **76.3** | 53.7  |
| OFA-X               | 30.1  | 22.5  | 10.9  | 9.2   | 17.6  | 25.4  | 25.7  | 3.9  | 68.9  | 50.5  |
| ReRe                | 34.0  | 21.2  | 13.8  | 9.0   | **20.8** | 29.4  | 35.5  | 4.2  | 74.9  | 47.5  |
| **VMARC-QA (ours)** | 27.5  | 14.8  | 8.1   | 4.4   | 17.6  | 22.4  | 23.6  | 4.0  | 76.0  | **64.8** |

## 📁 Repository Structure

The project is organized as follows:

```
.
├── src/
│   ├── agents/          # Contains the logic for each agent (Junior, Senior, Manager).
│   ├── core/            # Implements the core multi-agent graph using LangGraph.
│   ├── models/          # Defines Pydantic models for state management.
│   ├── tools/           # Houses tools for VQA and external knowledge retrieval.
│   ├── utils/           # Includes utility functions and helper scripts.
│   ├── evaluation/      # Scripts for evaluating model and agent performance.
│   └── main.py          # Entry point for running the application.
│
├── ViVQA/               # Submodule containing the underlying Vietnamese VQA model implementation.
│
├── notebooks/           # Jupyter notebooks for experimentation and analysis.
│
├── .env.example         # Example environment file.
└── requirements.txt     # Project dependencies.
```

## 📜 Citation

If this project is based on a research paper, add the BibTeX citation information here.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 
