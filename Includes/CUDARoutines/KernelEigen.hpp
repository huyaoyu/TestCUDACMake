#ifndef __CUDAROUTINES_KERNELEIGEN_HPP__
#define __CUDAROUTINES_KERNELEIGEN_HPP__

#include "CUDARoutines/CUDACommon.hpp"

int crExponent(const CRReal* input, int rows, int cols, int cStep, CRReal* output);

#endif // __CUDAROUTINES_KERNELEIGEN_HPP__