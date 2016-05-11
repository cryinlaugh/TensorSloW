#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "blob.h"

void read_mnist_images(char* fileName, Blob * inputData);

void __read_mnist_images(Tensor * inputData, char* fileName);

#endif  // CAFFE_BLOB_HPP_
