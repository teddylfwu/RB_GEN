#ifndef RANDOM_H
#define RANDOM_H

#include "common.h"

#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif


void InitRandom(int seed);

void UniformRandom(double *a, size_t n, double *high);

void StandardNormal(double *a, size_t n, double sigma);

void StudentT1(double *a, size_t n, double sigma);

void MultivariateStudentT1(double *a, size_t d, size_t r, double sigma);


#endif // RANDOM_H
