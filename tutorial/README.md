# Computer Vision: Deep Dive into Segmentation Approaches

## Setup
There are a lot of dependencies that break if they are mixed with newer
packages, so it's easiest to create a new miniconda enviroment and install
everything fresh here. You only need to do these steps for Chapter-2 (which is
run locally) and Chapters 3 and 4 (if you run locally rather than using the
[Colab tutorial][colab_tutorial].

### Miniconda
[Miniconda][miniconda] is an environment manager for Python. It's similar to Anaconda,
but we recommend Miniconda to save disk space and also to prevent any GUIs and excess
packages from being installed. Once you've download the miniconda package for
your platform, follow the [installation guide][installation].

__Note:__ You will need to run Python 3.6 or higher for all of our code to run.
If you install miniconda, it will install a compatible version by default.

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
[colab_tutorial]: https://github.com/mosdragon/kdd2020/blob/master/tutorial/Colab_Chapters_3_and_4/Training_Colab.ipynb
