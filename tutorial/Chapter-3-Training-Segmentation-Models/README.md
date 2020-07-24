# Training MMDetection

## Setup
In addition to the steps shown in the README under `tutorials`, we'll need to
install [mmdetection][mmdetection]. This package has many dependencies and is
very platform-dependent, but we'll write down the most simplified version of the
installation, and point you to platform-specific instructions.

Additionally, we've cloned the repository here to make installation faster.

### Install mmdetection
__Note:__ Run the following if you do not have CUDA on your machine (i.e if you
have macOS or Linux but without a CUDA-enabled GPU).
```bash
cd mmdetection

conda activate kdd_demo
conda install -y pytorch torchvision -c pytorch
pip install mmcv-full
pip install seaborn

pip install -v -e .
```

__If you have a CUDA-enabled GPU:__ See mmdetection's official [install
guide][install] to install PyTorch with CUDA for your machine.


---
[mmdetection]: https://github.com/open-mmlab/mmdetection
[install]: https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md
