#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray)  
{  
    if (LengthOfArray < 0)  
    {  
        return -1;  
    }  
    //int result = static_cast<signed int>(array[0]);  
    int result = (signed int)(array[0]);  
    for (int i = 1; i < LengthOfArray; i++)  
    {  
        result = (result << 8) + array[i];  
    }  
    return result;  
}  

void __read_mnist_images(Tensor * inputData, char const* fileName)
{
  FILE* ifs = fopen(fileName, "rb");
  unsigned char magic_number[4];
  unsigned char num_items[4];
  unsigned char num_rows[4];
  unsigned char num_cols[4];
  
  fread(&magic_number, 4, sizeof(char), ifs);
  fread(&num_items, 4, sizeof(char), ifs);
  fread(&num_rows, 4, sizeof(char), ifs);
  fread(&num_cols, 4, sizeof(char), ifs);

  int magic = ConvertCharArrayToInt(magic_number, 4);
  int items = ConvertCharArrayToInt(num_items, 4);
  int rows = ConvertCharArrayToInt(num_rows, 4);
  int cols = ConvertCharArrayToInt(num_cols, 4);

  printf("size of bin data : %d %d %d %d\n", magic, items, rows, cols);

  inputData->N = 1;
  inputData->C = cols;
  inputData->R = rows;
  inputData->B = items;
  inputData->size = 1*cols*rows*items;
  inputData->data = (real*)malloc( REALSIZE*inputData->size);

  unsigned char * image  = (unsigned char*)malloc(sizeof(char) * rows * cols);
  for(int i = 0; i < items; ++i){
    fread(image , rows*cols, sizeof(unsigned char), ifs);
    for( int r =  0; r < rows; ++r){
      for( int c = 0; c < cols; ++c){
        inputData->data[(r*cols + c)+i*(rows*cols)] = (real)image[r*cols + c]/255;
      }
    }
  }
  free(image);
}


void read_mnist_images(char const* fileName, Blob * inputData)
{
  FILE* ifs = fopen(fileName, "rb");
  unsigned char magic_number[4];
  unsigned char num_items[4];
  unsigned char num_rows[4];
  unsigned char num_cols[4];
  
  fread(&magic_number, 4, sizeof(char), ifs);
  fread(&num_items, 4, sizeof(char), ifs);
  fread(&num_rows, 4, sizeof(char), ifs);
  fread(&num_cols, 4, sizeof(char), ifs);

  int magic = ConvertCharArrayToInt(magic_number, 4);
  int items = ConvertCharArrayToInt(num_items, 4);
  int rows = ConvertCharArrayToInt(num_rows, 4);
  int cols = ConvertCharArrayToInt(num_cols, 4);

  printf("size of bin data : %d %d %d %d\n", magic, items, rows, cols);

  inputData->numChannel = 1;
  inputData->width = cols;
  inputData->height = rows;
  inputData->numImages = items;
  inputData->data = (float*)malloc( sizeof(float)*rows*cols*items );

  unsigned char * image  = (unsigned char*)malloc(sizeof(char) * rows * cols);
  for(int i = 0; i < items; ++i){
    fread(image , rows*cols, sizeof(unsigned char), ifs);
    for( int r =  0; r < rows; ++r){
      for( int c = 0; c < cols; ++c){
        inputData->data[(r*cols + c)+i*(rows*cols)] = (float)image[r*cols + c]/255;
      }
    }
  }

  free(image);
}


