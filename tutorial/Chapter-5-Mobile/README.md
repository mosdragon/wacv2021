# Converting Tensorflow Model to iOS-compatible MLModel
In this chapter, we demonstrate how to convert Tensorflow model to MLModel using a DeeplabV3+ Tensorflow model pre-trained on the ADE20K dataset. PyTorch models can also be converted to MLModels but the process is much more complicated and the conversion process uses [ONNX][onnx].


# Quantization
We also demonstrate how to reduce the size of MLModels using [Quantization][quantization].

# Setup
The last cell of this notebook where we test the inference of the converted MLModel needs to be run on MacOS 10.13 or later. If you only want to run the conversion and quantization, MacOS environment is not necessary.

---
[onnx]: https://mmdetection.readthedocs.io/en/latest/useful_tools.html#mmdetection-model-to-onnx-experimental
[quantization]: https://coremltools.readme.io/docs/quantization