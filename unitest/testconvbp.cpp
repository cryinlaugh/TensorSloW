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

    int inR = 5;
    int inC = 5;
    int inN = 3;
    int inB = 2;

    int filterDim = 3;

    int outR = inR + filterDim - 1;
    int outC = inC + filterDim - 1;
    int outN = 3;
    int outB = inB;

	srand( (unsigned)time( NULL ) );

    Tensor inputData, outputData;
    __TensorDataInit(&inputData, inR, inC, inN, inB);
    __TensorDataInit(&outputData, outR, outC, outN, outB);
    __TensorLoadData(&inputData, "../testdata/convbp/input.txt");
    //load data

    convLayer C1;
    C1.setUp( &inputData, &outputData, filterDim, filterDim);
    C1.setWeightFromFile("../testdata/convbp/W.txt","../testdata/convbp/b.txt");
    printf("setup is ok\n");
    C1.backward();
    //C1.checkBp(); //TODO


    return 0;
}
