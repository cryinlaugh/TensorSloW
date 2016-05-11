#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "time.h"

extern void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);

int main(int argc, char* argv[])
{
  int i;
  printf("test!\n");

  float A[6] = {1,2,3,4,5,6};
  float B[8] = {1,2,3,4,5,6,7,8};
  float C[12] = {1,2,3,4,5,6,7,8,9,0,1};
  int m = 3;
  int k = 2;
  int n = 4;
  
  char ta = 'T';
  char tb = 'N';
  float alpha = 1.0;
  float beta = 0.0;

  sgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &k, B, &k, &beta, C, &m);

  for(i=0; i<12; ++i)
    printf("%f\n", C[i]);

  return 0;
}
