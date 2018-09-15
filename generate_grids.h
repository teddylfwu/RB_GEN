#ifndef GRIDS_H
#define GRIDS_H

#include "common.h"
#include "random.h"

typedef enum {
  Gaussian=0,
  Laplace=1,
  ProdLaplace=2
} KernelType;

int GenerateGridParas(double *delta, double *mu, size_t d, size_t R, double sigma, KernelType kernel, int seed);

#endif // GRIDS_H
