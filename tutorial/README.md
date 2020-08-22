# Computer Vision: Deep Dive into Segmentation Approaches

## Pre-requisite
The minimum pre-requisite for following this tutorial is a programming environment
with access to basic command line tools. Ideally, a virtual machine or a local Linux
environment would work best.

__Note__: For chapters 3-4 you will find two separate versions of the tutorial. If you 
do not have access to a GPU, you can follow using the Google Colab version of our 
chapter 3-4 tutorials, which we will be presenting live. If you have access to a 
GPU-enabled environment, our recorded videos lead you through that version of the tutorial.

## Setup
There are a lot of dependencies that break if they are mixed with newer
packages, so it's easiest to create a new miniconda enviroment and install
everything fresh here.

### Miniconda
[Miniconda][miniconda] is an environment manager for Python. It's similar to Anaconda,
but we recommend Miniconda to save disk space and also to prevent any GUIs and excess
packages from being installed. Once you've download the miniconda package for
your platform, follow the installation guide [here][installation].

__Note:__ You will need to run Python 3.6 or higher for all of our code to run.
If you install miniconda, it will install a compatible version.

```bash
conda create --name kdd_demo
conda activate kdd_demo

pip install --upgrade numpy==1.16
pip install matplotlib imageio scipy pandas Pillow scikit-image imageio
pip install Cython
pip install pycocotools

conda install -y opencv
conda install -y jupyter
```

### Downloading Datasets
To use the notebooks, you will need some datasets downloaded into the
`../datasets` directory.

__ADE20K Dataset:__ This dataset is used in Chapter 2 to generate the datasets
we use in Chapters 3 and 4. To download, you can go to the [dataset
page][ade20k] and save to the datasets folder, or simply run the following
command in your terminal:
```bash
cd ../datasets/
wget https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
unzip -qq ADE20K_2016_07_26.zip
```

__Bedroom Scenes Datasets:__ These are the datasets Chapter 2 generates. If you
wish to download them instead of running the notebooks, you can download them
from [here][bedroom_voc] and [here][bedroom_coco], or again run the following:
```bash
cd ../datasets/

wget https://storage.googleapis.com/kdd2020hdvisai/static/datasets/bedroom_scenes_voc.zip
wget https://storage.googleapis.com/kdd2020hdvisai/static/datasets/bedroom_scenes_coco.zip

unzip -qq bedroom_scenes_voc.zip
unzip -qq bedroom_scenes_coco.zip
```

## Running the Notebooks
To run the notebooks, you must activate the `kdd_demo` conda environment and
start the Jupyter notebook server.
```bash
# Make sure you're in the tutorials directory.
cd <PROJECT-ROOT>/tutorials

conda activate kdd_demo
jupyter notebook
```

The command above will launch a browser window with Jupyer. You can then proceed
to whichever Chapter you'd like, and if all packages and datasets are properly
downloaded and installed, you can step through and run each notebook cell
one-by-one.

---
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[installation]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[ade20k]: https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
[bedroom_voc]: https://storage.googleapis.com/kdd2020hdvisai/static/datasets/bedroom_scenes_voc.zip
[bedroom_coco]: https://storage.googleapis.com/kdd2020hdvisai/static/datasets/bedroom_scenes_coco.zip
