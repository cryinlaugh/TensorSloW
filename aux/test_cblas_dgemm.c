#include <cblas.h>
#include <stdio.h>

int main()
{
  int i=0;
  double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};         
  double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};  
  double C[9] = {1,2,3,4,5,6,7,8,9}; 
  int m = 2;
  int k = 3;
  int n = 2;
//  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, A, m, B, k, 0.0, C, m);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, m, B, n, 0.0, C, n);

  for(i=0; i< m * n; i++)
    printf("%lf ", C[i]);
  printf("\n");


  return 0;
}
