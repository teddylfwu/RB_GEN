#ifndef WRITESVMFILE_H
#define WRITESVMFILE_H

#include "common.h"

int writeDEN2SVMfile(const char *filename, double *X, double *y, size_t n, size_t d);

int writeCSR2SVMfile(const char *filename, Mat *X, double *y); 

int writeGRID2file(const char *filename, double *delta, double *mu, size_t *offset, int *codeTable, size_t R, size_t d, size_t D);


#endif // WRITESVMFILE_H 
