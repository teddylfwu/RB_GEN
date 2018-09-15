#ifndef COMMON_H
#define COMMON_H

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#ifndef MAX
#define MAX(a,b) (((a)<(b))?(b):(a))
#endif
#ifndef MIN
#define MIN(a,b) (((a)>(b))?(b):(a))
#endif

#ifdef USE_MATLAB
# include "mex.h"
# define RB_INDEX mwIndex
# define RB_VAL double
# define MALLOC mxMalloc
# define CALLOC mxCalloc
# define REALLOC mxRealloc
# include <inttypes.h>
# define SIZET uint64_t
#else
# define RB_INDEX size_t
# define RB_VAL double
# define MALLOC malloc
# define CALLOC calloc
# define REALLOC realloc
# define SIZET size_t
#endif

enum Mat_type { Mat_type_dense=0, Mat_type_CSR=1 };
typedef struct {
  enum Mat_type type;
  size_t n; // number of data points
  size_t d; // number of data dimension
  size_t nnz; // number of nonzero elements in data matrix
  RB_VAL *val; 
  RB_INDEX *rowIndex;
  RB_INDEX *colIndex;
} Mat; // for data matrix

void PrintFPMatrix(char *name, double *a, size_t m, size_t n);

void PrintFPCOOMatrix(char *name, double *a, size_t *colIndex, size_t m, size_t n);

void PrintIntMatrix(char *name, int *a, size_t m, size_t n);

void PrintSizetMatrix(char *name, size_t *a, size_t m, size_t n);

void PrintIntCOOMatrix(char *name, size_t *a, size_t *colIndex, size_t m, size_t n);

#ifdef _OPENMP
#include <omp.h>
#define CLOCK omp_get_wtime()
#else
#define CLOCK ((double)clock()/CLOCKS_PER_SEC)
#endif

#endif // COMMON_H
