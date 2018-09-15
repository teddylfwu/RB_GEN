#include <cmath>
#include <cstring>
#include <cstdlib>
#include <complex>
#include <cassert>
#include "mex.h"

#define USE_MATLAB
#include "../feature_matrix.c"

// Main function: dispatch the call to the proper mexFunction_* function
// Input: A, transposed matrix; double; delta: matrix; mu: matrix
// Output: offset: array; codeTable: matrix; Phi: sparse matrix transposed
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check arguments

    if (nrhs != 3) {
        mexErrMsgTxt("Invalid number of input arguments, they should be 3");
    }
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Invalid argument 1: only non-complex matrices are supported");
    }
    if (!mxIsNumeric(prhs[1]) || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("Invalid argument 2: only non-complex matrices are supported");
    }
    if (!mxIsNumeric(prhs[2]) || mxIsComplex(prhs[2])) {
        mexErrMsgTxt("Invalid argument 3: only non-complex matrices are supported");
    }
    if (mxGetM(prhs[0]) != mxGetM(prhs[1])) {
        mexErrMsgTxt("Invalid argument 2: it should have as many rows as rows have argument 1");
    }
    if (mxGetM(prhs[0]) != mxGetM(prhs[2])) {
        mexErrMsgTxt("Invalid argument 3: it should have as many rows as rows have argument 1");
    }
    if (mxGetN(prhs[1]) != mxGetN(prhs[2])) {
        mexErrMsgTxt("Invalid argument 3: it should have as many columns as columns have argument 2");
    }
    if (nlhs < 2 || nlhs > 3) {
        mexErrMsgTxt("Invalid number of output arguments, they should be 2 or 3");
    }
 
    int R = mxGetN(prhs[2]);
    Mat X;
    if (mxIsSparse(prhs[0])) {
        // NOTE: MATLAB stores sparse matrices in CSC but this code uses CSR.
        //       The solution is that the caller should pass the matrix
        //       transposed.
        X.type = Mat_type_CSR;
        X.rowIndex = mxGetJc(prhs[0]);
        X.colIndex = mxGetIr(prhs[0]);
        X.val = (RB_VAL*)mxGetPr(prhs[0]);
        X.d = mxGetM(prhs[0]);
        X.n = mxGetN(prhs[0]);
        X.nnz = X.rowIndex[X.n];
    } else {
        // NOTE: MATLAB stores dense matrices in column major but this code
        //       uses row major. The solution is that the caller should pass
        //       the matrix transposed.
        X.type = Mat_type_dense;
        X.rowIndex = NULL;
        X.colIndex = NULL;
        X.val = (RB_VAL*)mxGetData(prhs[0]);
        X.n = mxGetN(prhs[0]);
        X.d = mxGetM(prhs[0]);
        X.nnz = X.n*X.d;
    }

    double *delta = (double*)mxGetData(prhs[1]); 
    double *mu = (double*)mxGetData(prhs[2]); 
    SIZET *offset = (SIZET*)mxMalloc(sizeof(SIZET)*(R+1));
    int *codeTable = NULL;
    
    Mat Phi;
    if (nlhs == 3) {
        Phi.type = Mat_type_CSR;
        Phi.rowIndex = (RB_INDEX*)mxMalloc(sizeof(RB_INDEX)*(X.n+1));
        Phi.colIndex = (RB_INDEX*)mxMalloc(sizeof(RB_INDEX)*(X.n*R));
        Phi.val = (double*)mxMalloc(sizeof(double)*(X.n*R));
        Phi.d = 0;
        Phi.n = X.n;
        Phi.nnz = X.n*R;
    }
    ComputeTrainFeatureMatrix(R, X, delta, mu, offset, &codeTable, nlhs == 3?&Phi:(Mat*)NULL);

    // Create MATLAB offset, codeTable and Phi
    plhs[0] = mxCreateNumericArray(0, 0, mxUINT64_CLASS, mxREAL);
    mxFree(mxGetData(plhs[0]));
    mxSetData(plhs[0], offset);
    mxSetM(plhs[0], (mwSize)R+1);
    mxSetN(plhs[0], (mwSize)1);
    
    plhs[1] = mxCreateNumericArray(0, 0, mxINT32_CLASS, mxREAL);
    mxFree(mxGetData(plhs[1]));
    mxSetData(plhs[1], codeTable);
    mxSetN(plhs[1], (mwSize)offset[R]);
    mxSetM(plhs[1], (mwSize)X.d);

    if (nlhs == 3) {
        // It is created transposed!!!
        plhs[2] = mxCreateSparse(0, 0, 0, mxREAL);
        mxFree(mxGetPr(plhs[2]));
        mxSetPr(plhs[2], Phi.val);
        mxFree(mxGetIr(plhs[2]));
        mxSetIr(plhs[2], Phi.colIndex);
        mxFree(mxGetJc(plhs[2]));
        mxSetJc(plhs[2], Phi.rowIndex);
        mxSetM(plhs[2], offset[R]);
        mxSetN(plhs[2], X.n);
        mxSetNzmax(plhs[2], Phi.nnz);
    }

    // Use MATLAB offsets
    for (int i=0; i<=R; i++) offset[i] += 1;
}
