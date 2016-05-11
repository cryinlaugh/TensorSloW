extern "C"{
#include "blob.h"
#include "cnnPoolingImp.h"
}

class poolingLayer{
public:
	Blob convFeatures;

	void cnnPool(int const poolRow, int const poolCol, Features const convFeatures, char* pooltype, Features* pooledFeatures, float* weight);

	Tensor* input, *output; 
	int _poolRow, _poolCol;
	real weight;

	void setUp(Tensor* in, Tensor* out, int poolRow, int poolCol){
		_poolRow = poolRow;
		_poolCol = poolCol;

		input = in;
		output = out;
	}

	void forward(){
		__forwardPooling(input, output, &weight, _poolRow, _poolCol, "max");
		__TensorPrint(output, "./log/pooledData.txt");
	}
};
