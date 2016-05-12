#include "cnnPoolingImp.h"

void __forwardPooling(Tensor* const input, Tensor* const output, real* W, int poolRow, int poolCol, char * const pooltype){
	int B = input->B;
	int N = input->N;
	int R = input->R;
	int C = input->C;
	int pooledDimRow = output->R;//(R+poolRow-1) / poolRow;
	int pooledDimCol = output->C; //(convolvedDimCol+poolCol-1) / poolCol;

	*W = (real)1/poolRow/poolCol;

	for (int img = 0; img < B; ++img){
		for (int fea = 0; fea < N; ++fea)
				for(int j = 0; j < pooledDimRow; ++j){
					for(int i = 0; i < pooledDimCol; ++i){
					float sum = 0.0;
					for (int jj = poolRow*(j); jj<(j+1)*poolRow && jj<R; ++jj)
						for(int ii = poolCol*(i); ii<(i+1)*poolCol && ii<C; ++ii)
						{
							sum += *(input->data + ii + jj*R + 
								fea*R*C + img*N*C*R);
						}
					*(output->data+i+j*pooledDimCol+fea*pooledDimRow*pooledDimCol + img*pooledDimCol*pooledDimRow*N)=
						sum/poolRow/poolCol;
					}
				}
		}
}