#include "innerprodImp.h"

void __innerprod_forward(Tensor* input, Tensor* W, Tensor* output){
	int m = W->K2;
	int k = W->K1;
	int n = input->B;

	char ta = 'N';
	char tb = 'N';
	real alpha = 1.0;
	real beta = 0.0;

	printf("in innerprod_forward before sgemm, W: %d, input: %d\n", W->size, input->size);
	sgemm_(&ta, &tb, &m, &n, &k, &alpha, W->data, &k, input->data, &k, &beta, output->data, &m);		
}
