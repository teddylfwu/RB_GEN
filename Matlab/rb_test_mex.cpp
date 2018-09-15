#include <cmath>
#include <cstring>
#include <cstdlib>
#include <complex>
#include <cassert>
#include "mex.h"

#define USE_MATLAB
#include "../feature_matrix.c"

// Main function: dispatch the call to the proper mexFunction_* function
// Input: A, transposed matrix; double; delta: matrix; mu: matrix,
//        offset: array; codeTable: matrix
// Output: Phi: sparse matrix transposed
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check arguments

    if (nrhs != 5) {
        mexErrMsgTxt("Invalid number of input arguments, they should be 5");
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

    int R = mxGetN(prhs[1]);

    if (mxGetM(prhs[0]) != mxGetM(prhs[1])) {
        mexErrMsgTxt("Invalid argument 2: it should have as many rows as rows have argument 1");
    }
    if (mxGetM(prhs[0]) != mxGetM(prhs[2])) {
        mexErrMsgTxt("Invalid argument 3: it should have as many rows as rows have argument 1");
    }
    if (mxGetN(prhs[1]) != mxGetN(prhs[2])) {
        mexErrMsgTxt("Invalid argument 3: it should have as many columns as columns have argument 2");
    }
    if (!mxIsUint64(prhs[3]) || mxGetN(prhs[3]) != 1 || mxGetM(prhs[3]) != mxGetN(prhs[1])+1) {
        mexErrMsgTxt("Invalid argument 4: it should be a vector of class uint64 of length the number of columns in argument 2 plus one");
    }
    SIZET *offset = (SIZET*)mxGetData(prhs[3]);

    if (!mxIsInt32(prhs[4]) || mxGetM(prhs[4]) != mxGetM(prhs[0]) || mxGetN(prhs[4]) != offset[R] - offset[0]) {
        mexErrMsgTxt("Invalid argument 5: it should be a vector of class int32 with as many rows as the number of columns in argument 1 and as many columns as the difference between the last and the first value of the array of argument 4");
        }
    if (nlhs != 1) {
        mexErrMsgTxt("Invalid number of output arguments, they should be 1");
    }
 
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
    int *codeTable = (int*)mxGetData(prhs[4]);
    
    // Remove +1 from MATLAB offsets
    for (int i=0; i<=R; i++) offset[i] -= 1;

    Mat Phi;
    Phi.type = Mat_type_CSR;
    Phi.rowIndex = (RB_INDEX*)mxMalloc(sizeof(RB_INDEX)*(X.n+1));
    Phi.colIndex = (RB_INDEX*)mxMalloc(sizeof(RB_INDEX)*(X.n*R));
    Phi.val = (double*)mxMalloc(sizeof(double)*(X.n*R));
    Phi.d = 0;
    Phi.n = X.n;
    Phi.nnz = X.n*R;

    ComputeTestFeatureMatrix(R, X, delta, mu, offset, codeTable, &Phi);

    // Create MATLAB Phi
    // It is created transposed!!!
    plhs[0] = mxCreateSparse(0, 0, 0, mxREAL);
    mxFree(mxGetPr(plhs[0]));
    mxSetPr(plhs[0], Phi.val);
    mxFree(mxGetIr(plhs[0]));
    mxSetIr(plhs[0], Phi.colIndex);
    mxFree(mxGetJc(plhs[0]));
    mxSetJc(plhs[0], Phi.rowIndex);
    mxSetM(plhs[0], offset[R]);
    mxSetN(plhs[0], X.n);
    mxSetNzmax(plhs[0], Phi.nnz);

    // Use MATLAB offsets
    for (int i=0; i<=R; i++) offset[i] += 1;
}
