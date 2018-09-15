#ifndef FEATURE_MATRIX_H
#define FEATURE_MATRIX_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

int ComputeTrainFeatureMatrix(size_t R, Mat X, double *delta, double *mu, SIZET *offset, int **codexTable, Mat *Phi);

int ComputeTestFeatureMatrix(size_t R, Mat X, double *delta, double *mu, SIZET *offset, int *codeTable, Mat *Phi);

void ComputeBinNum(Mat X, size_t row, int *code, double *delta, double *mu);

int compareCode(int *code, int *codeTable, size_t start, size_t end, size_t d, size_t *nonEmptyBinsCount);

void copyCodeTocodeTable(int *code, int *codeTableAppend, size_t d);

void copycodeTableLocalTocodeTable(int *codeTableLocal, int *codeTable, size_t end, size_t d);

#ifdef __cplusplus
}
#endif

#endif // FEATURE_MATRIX_H
