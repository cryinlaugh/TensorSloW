#ifndef BLOB_H_
#define BLOB_H_
#include <stdio.h>
#include <string.h>

typedef float real;
#define REALSIZE sizeof(real)

typedef struct{
  int numChannel;
  int width;
  int height;
  int numImages; //filter number
  float * data;
}Blob;

typedef struct{
	int filterDim;
	int channel;
	int numFilters;
	float * data;
}Weight;

typedef struct{
	int imageRow;
	int imageCol;
	int numFeatures;
	int numImages;
	float * data;
}Features;

typedef struct{
	int R, C, N, B; //for images, W
	int K1, K2, Ni, No; //for filter
	int size;
	real* data;
}Tensor;

void __TensorLoadData(Tensor* T, char const* filename);

void __TensorLoadB(Tensor* b, char const* filename);

void __TensorLoadWeight(Tensor* W, char const* filename);

void __TensorCopy(Tensor* TA, Tensor* TB, int numImg);

void __TensorDataInit(Tensor* T, int R, int C, int N, int B);
void __TensorDataInitRandom(Tensor* T, real lower_bound, real upper_bound);

void __TensorPrint(Tensor* T, char const* filename);

void __TensorCheckRes(char const* fn1, char const* fn2);

//void __TensorPrint(Tensor* const T, char* const filename);

#endif
