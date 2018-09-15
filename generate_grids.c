#include "generate_grids.h"

int GenerateGridParas(double *delta, double *mu, size_t d, size_t R, double sigma, KernelType kernel, int seed) {

  InitRandom(seed); 
  switch (kernel) {
    case Gaussian:
      StandardNormal(delta, d*R, sigma);
      break;
    case Laplace:
      MultivariateStudentT1(delta, d, R, sigma);
      break;
    case ProdLaplace:
      StudentT1(delta, d*R, sigma);
      break;
  }
  UniformRandom(mu, d*R, delta);

  return 0;

}
