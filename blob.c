#include "blob.h"
#include <math.h>

void __TensorLoadData(Tensor* T, char* filename){
	FILE * fh = fopen(filename, "r");
	if(!fh)
		printf("no data file!\n");
	for(int i=0; i<T->size; ++i){
		fscanf(fh, "%f", T->data+i);
	}
	fclose(fh);
}

void __TensorLoadB(Tensor* b, char* filename){
	printf("bias size is %d\n", b->N);
	FILE * fh = fopen(filename, "r");
	if(!fh)
		printf("no bais file!\n");
	for(int i=0; i<b->N; ++i){
		fscanf(fh, "%f", b->data+i);
	}
	fclose(fh);
}

void __TensorLoadWeight(Tensor* W, char* filename){

	int size = W->size;
	W->data = (real*)malloc(REALSIZE*size);

	FILE * fh = fopen(filename, "r");
	if(!fh){
		printf("no Weight file!\n");
		exit(0);
	}
	for(int i=0; i<size; ++i){
		fscanf(fh, "%f", W->data+i);
	}
	fclose(fh);
}

void __TensorCopy(Tensor* TA, Tensor* TB, int numImg){
	TB->N = TA->N;
	TB->R = TA->R;
	TB->C = TA->C;
	TB->B = numImg;
	TB->size = TB->B * TB->C * TB->R * TB->N;
	//from start;
	TB->data = TA->data;

	TB->data = (real* )malloc(REALSIZE*TB->size);
	TB->data = memcpy(TB->data, TA->data, REALSIZE* TB->size);
	printf("copy is ok\n");
}

void __TensorDataInit(Tensor* T, int R, int C, int N, int B){
	T->R = R; T->C = C; T->N = N; T->B = B; T->size = R*C*N*B;
	T->data = (real*)malloc(REALSIZE*R*C*N*B);
}

void __TensorPrint(Tensor* const T, char* filename){
	int size = T->size;

	FILE * fh = fopen(filename, "w");
	for(int i = 0; i<size; ++i){
		fprintf(fh, "%.6f\n", T->data[i] );
	}
	fclose(fh);
}

void __TensorCheckRes(char* fn1, char* fn2){
	FILE * f1 = fopen(fn1, "r");
	FILE * f2 = fopen(fn2, "r");

	int pos = 0;
	float x,y;
	printf("Begin check\n");
	while(fscanf(f1, "%f", &x)==1 && fscanf(f2, "%f", &y)==1){
		if(fabsf(x-y) > 1e-3){
			printf("%f %f Error check @ line %d!\n", x, y, pos);
			break;
		}
		pos++;
	}
	printf("Check OK!\n");
	fclose(f1);
	fclose(f2);
}
