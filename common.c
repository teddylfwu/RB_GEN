#include "common.h"

// print dense matrix with real numbers in row major
void PrintFPMatrix(char *name, double *a, size_t m, size_t n){

  printf("%s:\n", name);
  size_t i,j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%.16g ", a[i*n + j]);
    }
    printf("\n");
  }
  printf("\n");

}


// print sparse matrix (COO format) with real numbers in row major
void PrintFPCOOMatrix(char *name, double *a, size_t *colIndex, size_t m, size_t n){

  printf("%s:\n", name);
  size_t cnz = 0;
  size_t i,j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      if (colIndex[cnz] == j){
        printf("%.16g ", a[cnz]);
        cnz++;
      }
    }
    printf("\n");
  }
  printf("\n");

}


// print dense matrix with integer numbers in row major
void PrintIntMatrix(char *name, int *a, size_t m, size_t n){

  printf("%s:\n", name);
  size_t i,j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%d ", a[i*n + j]);
    }
    printf("\n");
  }
  printf("\n");

}


// print dense matrix with size_t in row major
void PrintSizetMatrix(char *name, size_t *a, size_t m, size_t n){

  printf("%s:\n", name);
  size_t i,j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%zu ", a[i*n + j]);
    }
    printf("\n");
  }
  printf("\n");

}


// print sparse matrix (COO) with integer numbers in row major
void PrintIntCOOMatrix(char *name, size_t *a, size_t *colIndex, size_t m, size_t n){

  printf("%s:\n", name);
  size_t cnz = 0;
  size_t i,j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      if (colIndex[cnz] == j){
        printf("%zu ", a[cnz]);
        cnz++;
      }
    }
    printf("\n");
  }
  printf("\n");

}
