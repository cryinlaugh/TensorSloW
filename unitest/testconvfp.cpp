extern "C"{
    #include "mnist.h"
    #include "blob.h"
}
#include "convLayer.hpp"
#include "cnnPool.hpp"
#include <stdlib.h>
#include <time.h>

int main()
{

	int imageDim = 28;

	int filterDim = 8;
	int numFilters = 3;
	int numImages = 60000;
	int poolDim = 3;

	srand( (unsigned)time( NULL ) );

    Tensor inputData;
    __read_mnist_images(&inputData,"../../data/train-images.idx3-ubyte");
    Tensor inData; //28*28*1*8
    __TensorCopy(&inputData, &inData , 8);


    convLayer C1;
    Tensor C1Data; //21*21*3*8
    __TensorDataInit(&C1Data,21,21,3,8);
    C1.setUp( &inData, &C1Data, filterDim, filterDim);
    C1.setWeightFromFile("../testdata/convfp/W.txt","../testdata/convfp/b.txt");
    C1.forward(0,1);


    Tensor P1Data;
    __TensorDataInit(&P1Data, 11, 11, 3, 8);
    poolingLayer P1;
    P1.setUp( &C1Data, &P1Data, 2, 2);
    P1.forward();

    return 0;
}
