extern "C"{
	#include "Blob.h"
}
class innerprodLayer{
public:
	Tensor* input, *output; //inputDim*batchNum outpuDim*batchNUm
	Tensor W; //outputDim * inputDim; 
	Tensor b; //outputDim * batchNum;


	void setUp(Tensor* in, Tensor*out){
		int inputDim = W->K1;
		int outputDim = W->K2;
		int B = input->B;

		W.size = inputDim*outputDim;
		W.K1 = inputDim;
		W.K2 = outputDim;
		W.data = (real*)malloc(REALSIZE*W.size);

		b.size = outputDim;
		b.data = (real*)malloc(REALSIZE*b.size);

		input = in;
		output = out;
	}
	void forward(){
		
	}

	void backward(){

	}


}