# Debiased fact-checking model

This code is based on ["Counterfactual VQA: A Cause-Effect Look at Language Bias"](https://github.com/yuleiniu/cfvqa)

## Installation

It is recommended to install Python 3.8.0

Install all required Python packages using:
```bash
conda create -n tor113 python==3.8.0
conda activate tor113
pip install torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl
pip install torchaudio-0.13.1+cu117-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.14.1+cu117-cp38-cp38-linux_x86_64.whl

pip install -r requirements.txt
```