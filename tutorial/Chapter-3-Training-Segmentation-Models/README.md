# Training MMDetection

## Setup
In addition to the steps shown in the README under `tutorials`, we'll need to
install [mmdetection][mmdetection]. This package has many dependencies and is
very platform-dependent, but we'll write down the most simplified version of the
installation, and point you to platform-specific instructions.

Additionally, we've cloned the repository here to make installation faster.

### Install mmdetection
__Note:__: if you do not have CUDA on your machine (i.e if you
have macOS or Linus but without a CUDA-enabled GPU), please follow
the Colab version of our Chapter 3-4 tutorial, which still trains
using the same dataset we prepared in Chapter 2.

(obsolete): Run the following if you do not have CUDA on your machine (i.e if you
have macOS or Linux but without a CUDA-enabled GPU).
```bash
conda activate kdd_demo
conda install -y pytorch torchvision -c pytorch
pip install -U "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install mmcv-full
pip install seaborn
pip install torch

cd mmdetection
pip install -v -e .
```

__If you have a CUDA-enabled GPU:__ See mmdetection's official [install
guide][install] to install PyTorch with CUDA for your machine. The installation
line `conda install -y pytorch torchvision -c pytorch` will change slightly if
you have a CUDA-enabled device.


### Download Pretrained Model
Since model training is very compute-intensive and takes hours even with a GPU
present, we've trained models for 2-6 hours that we can share with you here.

To download the best models, run the following in your terminal from this
_tutorials_ directory:
```bash
cd mmdetection
mkdir -p work_dirs/ade20k/short
mkdir -p work_dirs/ade20k/long

cd work_dirs/ade20k/long
wget https://storage.googleapis.com/kdd2020hdvisai/static/models/work_dirs_ade20k_long.zip
unzip -qq work_dirs_ade20k_long.zip
```

---
[mmdetection]: https://github.com/open-mmlab/mmdetection
[install]: https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md
