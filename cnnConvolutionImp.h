#ifndef _CNNCONVOLUTIONIMP_H_
#define _CNNCONVOLUTIONIMP_H_

#include "blob.h"

void __forward_im2col(Tensor* const in, Tensor* const W, Tensor* col_data);
void __backward_im2col(Tensor* const in, Tensor* const W, Tensor* col_data_full);

void __convForward(Tensor* const col_data, Tensor* const Weight, Tensor* const b, Tensor* ConvData);
void __convBackward(Tensor* const col_data_full, Tensor* const Weight, Tensor* const b, Tensor* currError);

void __convForward2(Tensor* const col_data, Tensor* const Weight, Tensor* const b, Tensor* ConvData);
void __convBackward2(Tensor* const prevError, Tensor* const W, Tensor* currError);

#endif
