#ifndef READFILE_H
#define READFILE_H

#include "common.h"

int readSVMfile(const char *filename, Mat *X, double **y);

int VerifyFileFormat(const char *filename, size_t *d, size_t *n, size_t *nnz);

int readGRIDfile(const char *filename, double **delta, double **mu, size_t **offset, int **codeTable, size_t *R, size_t d, size_t *D); 


#endif // READFILE_H
