extern "C"{
#include "mnist.h"
#include "blob.h"
#include "sys/time.h"
#include "time.h"

}
#include "convLayer.hpp"
#include "cnnPool.hpp"
#include <stdlib.h>


//test performance of 1 conv layer (forward only)
int main()
{
    
    int Ni = 64;
    int Ri = 32;
    int Ci = 32;
    int K = 4;
    int No = 64;
    int Ro = Ri-K+1;
    int Co = Ci-K+1;
    int B = 2;
    
    //Total FLOP:
    long long flop = 2*No*Ro*Co*K*K*Ni*B;
	
		printf("flop = %ld \n", flop);
	
    struct timeval start,finish;
    float duration;

    printf("[INFO]Parameters:Ni=%d, Ri=Ci=%d, K=%d, No=%d, Ro=Co=%d, B=%d\n",
           Ni, Ri, K, No, Ro, B);
    printf("[INFO]Init input Tensors\n");
    //init input 4d Tensor (Ri, Ci, Ni, B)
    Tensor inputData;
    __TensorDataInit(&inputData, Ri, Ci, Ni, B );
    __TensorDataInitRandom(&inputData, 0, 1);
    
    Tensor outputData;
    __TensorDataInit(&outputData, Ro, Co, No, B);
    
    
    printf("[INFO]Init Weights Tensors\n");
    convLayer C1;
    C1.setUp( &inputData, &outputData, K, K);
    C1.initWeightRandom();
    
    printf("[INFO]Run Forward\n");
    gettimeofday(&start, NULL);
    C1.forward(0,1);
    gettimeofday(&finish, NULL);
    duration = ((float)(finish.tv_sec-start.tv_sec)*1000000 + (float)(finish.tv_usec-start.tv_usec)) / 1000000;
    printf("[INFO]Finish Forward.\n");
    printf("[INFO]Time: %.6f sec\n", duration);
    printf("[INFO]FLOP: %.6f GFLOP\n", (float)flop/1000000000);
    printf("[INFO]Perf: %.6f GFLOPS\n", (float)flop/1000000000/duration);
    
    return 0;
}
