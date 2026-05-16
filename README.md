# Enduring Heterogeneous Cooperative Embodied Vision-Language Model for Long-Sequence Warehouse Tasks

## Overview

This work presents a **heterogeneous collaborative embodied vision-language framework** for long-sequence warehouse tasks, enabling stable and efficient multi-agent cooperation in dynamic parcel dispatching and sorting scenarios.

Key contributions:

- **Heterogeneous multi-agent system**: Two quadruped robots + one wheeled dual-arm manipulator for full-cycle sorting, transportation, and packing.
- **Three-stage reinforcement fine-tuning**: SFT → Dense RFT → **Variance-Suppressed Policy Optimization (VSPO)** for stable long-horizon learning.
- **VLM-integrated safety gate**: **ReAd (Reinforced Advantage Decision)** evaluates candidate actions before execution, improving safety and dispatch reliability.
- **Dynamic randomized environments**: Robust generalization across randomized layouts and parcel inflows.
s
## Quick Start

### Environment Setup

```bash
# Clone repository
git clone https://github.com/mikecoffee77/HeteroCollab-VIKI.git
cd HeteroCollab-VIKI

# Create Conda environment
conda env create -f hcviki.yml
conda activate hcviki
```

### Framework Installation

```bash
cd verl
pip install --no-deps -e .
cd ..

pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Data Preparation

```bash
# Download VIKI-R dataset from Hugging Face
git clone https://huggingface.co/datasets/henggg/VIKI-R
```

### Training

#### Step 1: Supervised Fine-Tuning (SFT)

```bash
# Prepare LLaMA-Factory environment
# Use https://github.com/hiyouga/LLaMA-Factory and place CoT data in dataset_info.json

# Train model with SFT
llamafactory-cli train configs/viki-1-3b.yaml
```

#### Step 2: Reinforcement Learning with VSPO-ReAd

```bash
cd train/3BGRPO/VIKI-L1

# Initialize VIKI-R-zero training
bash VIKI-R-zero.sh

# Start VIKI-R
bash VIKI-R-VSPO.sh
```

## Model Weights
L1 model checkpoints trained with our framework are available at:
[Qwen2.5VL-3B-Instruct-VIKI-R-VSPO-L1](https://huggingface.co/yjx8888/Qwen2.5VL-3B-Instruct-VIKI-R-VSPO-L1)