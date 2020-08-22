# Computer Vision: Deep Dive into Object Segmentation Approaches
This repo hosts the site and the resources for The Home Depot's Computer Vision
Computer Vision Workshop for KDD 2020.

![Segmentation Visualized][site/img/segmentation.png]

## Workshop Site
The website is currently hosted [here][site]. It contains information about the
workshop presenters as well as our abstract.

## Presenters
- Cody Wang
- Osama Sakhi
- Matthew Hagen
- Ala Eddine Ayadi

## Downloading the Repository
You will need the [Git][git] client in a terminal to download all the packages
and run the following command in your terminal:
```bash
git clone --recursive https://github.com/mosdragon/kdd2020.git
```

## Tutorial
This repo contains all resources used for the workshop, including our
Python-based tutorial, which you can execute on your own computer.

In this repo, you'll find two versions of our tutorial:
* _Jupyter notebook-only version_: This version requires a working Jupyter
  notebook installation and a CUDA-enabled GPU to run training and inference.
  This version can be found in the `tutorial` directory, with each notebook
  under a different directory in the form `Chapter-x`.
* _Google Colab Version_: This version runs on [Google Colab][colab], an online
  notebook hosted by Google with pre-installed packages and access to a
  CUDA-enabled GPU. This GPU will allow you to run our training and
  post-processing code without needing a GPU of your own. This can be found in
  `tutorial/Colab-Chapters_3_and_4`.

## Directory Structure
Here's the layout of our project. When you download the repository, your
directory structure will look exactly like this until you download and generate
new datasets.
```
.
├── README.md
├── datasets
│   ├── ADE20K_2016_07_26
│   ├── ADE20K_2016_07_26.zip
│   ├── README.md
│   ├── bedroom_scenes_coco
│   └── bedroom_scenes_voc
├── site
│   ├── Makefile
│   ├── css
│   ├── img
│   ├── index.html
│   ├── js
│   └── sass
└── tutorial
    ├── Chapter-1-Introduction
    ├── Chapter-2-Preprocessing
    ├── Chapter-3-Training-Segmentation-Models
    ├── Chapter-4-Postprocessing
    ├── Colab-Chapter3_4
    └── README.md
```

## Downloading Datasets
To run through Chapter 2, which generates the COCO and VOC-formatted datasets
from the original ADE20K dataset, you'll need to first download the full ADE20K
dataset. You can do so by running the following:
```bash
cd datasets
wget https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
unzip -qq ADE20K_2016_07_26.zip
```

If you want to skip Chapter 2 altogether and just move onto training and
post-processing, you can download the generated datasets by running the
following.
```bash
cd datasets
wget <TODO>
wget <TODO>

unzip -qq <TODO>
unzip -qq <TODO>
```

---

[site]: https://storage.googleapis.com/kdd2020hdvisai/static/index.html
[colab]: https://colab.research.google.com/notebooks/intro.ipynb
[git]: https://git-scm.com/downloads
