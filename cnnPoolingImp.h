#ifndef _CNNPOOLINGIMP_H_
#define _CNNPOOLINGIMP_H_

#include "blob.h"

void __forwardPooling(Tensor* const input, Tensor* const output, real* W, int poolRow, int poolCol, char const* pooltype);

#endif