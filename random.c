#include "random.h"

void InitRandom(int seed){

    srandom(seed);
}


void UniformRandom(double *a, size_t n, double *high){

    size_t i;
    for (i = 0; i < n; i++) {
        a[i] = (double)rand()/RAND_MAX;
    }

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; i++) {
        a[i] *= high[i];
    }

}


void StandardNormal(double *a, size_t n, double sigma){

    size_t i;
    for (i = 0; i < n/2; i++) {
        double U = (double)rand()/RAND_MAX;
        double V = (double)rand()/RAND_MAX;
        double common1 = sqrt(-2.0 * log(U));
        double common2 = 2.0 * M_PI * V;
        a[2 * i] = common1 * cos(common2);
        a[2 * i + 1] = common1 * sin(common2);
    }
    if (n % 2 == 1) {
        double U = (double)rand()/RAND_MAX;
        double V = (double)rand()/RAND_MAX;
        double common1 = sqrt(-2.0 * log(U));
        double common2 = 2.0 * M_PI * V;
        a[n - 1] = common1 * cos(common2);
    }

    double inv_sigma = 1.0/sigma;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; i++) {
        a[i] *= inv_sigma;
     }

}


void StudentT1(double *a, size_t n, double sigma){

    size_t i;
    for (i = 0; i < n; i++)
    {
        double V = (double)rand()/RAND_MAX;
        a[i] = tan(2.0 * M_PI * V);
        if (V > 0.5) {
          a[i] = -a[i];
        }
    }
    
    double inv_sigma = 1.0/sigma;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; i++) {
        a[i] *= inv_sigma;
     }
}


void MultivariateStudentT1(double *a, size_t d, size_t r, double sigma){

    size_t i,j;
    double *b = (double *)calloc(r, sizeof(double));
    assert(b != NULL);
    
    StandardNormal(a, d * r, sigma);
    StandardNormal(b, r, 1.0);

// matrix a is d by r, row major    
#ifdef _OPENMP
#pragma omp parallel for private(i,j)
#endif
    for (i = 0; i < r; i++) {
        for (j = 0; j < d; j++) {
            a[i * d + j] /= fabs(b[i]);
        }
    }
    
    free(b);

}
