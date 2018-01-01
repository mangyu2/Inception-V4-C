Inception-V4 Inference in C
==================================
A pure C version of Inception-V4 for inference
----------------------------------

This is a naive implementation of the inference of Inception V4 in pure C language. The behavior is exactly the same as the TensorFlow/GPU version. The inference time of a regular image is 3-4 minutes on i5-6500. Since this code is intended to be converted to HLS Sythesizable code, I didn't do any optimization for CPU computation also mini-batch processing is also not implemented, that's why I call it NAIVE version.
The model files can that the C code used can be extracted by model/extract.py from the checkpoint file acquired from http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
