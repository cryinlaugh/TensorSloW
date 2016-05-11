This is a CNN frameword for Sunway Cluster @ Wuxi

The Compilers on this cluster are not support C++ very well. So I write the
parts of core calculation in C and link them with the C++ parts.

The code is designed with as less dependencies as possible. Only a blas 
library is required!
