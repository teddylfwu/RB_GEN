#include <cmath>
#include <cstring>
#include <cstdlib>
#include <complex>
#include <cassert>
#include "mex.h"

#define USE_MATLAB
#include "../random.c"
#include "../generate_grids.c"

// Main function: dispatch the call to the proper mexFunction_* function
// Input: kernel: double; sigma: double; R: double; d: double; int seed
// Output: delta: matrix; mu: matrix
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check arguments

    if (nrhs != 4 && nrhs != 5) {
        mexErrMsgTxt("Invalid number of input arguments, they should be 4 or 5");
    }
    if (!mxIsScalar(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Invalid argument 1: it should be scalar");
    }
    int kernel = (int)mxGetScalar(prhs[0]);
    if (kernel < 0 || kernel > 2) {
        mexErrMsgTxt("Invalid argument 1: it should be 0, 1 or 2");
    }
    if (!mxIsScalar(prhs[1]) || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("Invalid argument 2: it should be scalar");
    }
    if (!mxIsScalar(prhs[2]) || mxIsComplex(prhs[2])) {
        mexErrMsgTxt("Invalid argument 3: it should be scalar");
    }
    if (!mxIsScalar(prhs[3]) || mxIsComplex(prhs[3])) {
        mexErrMsgTxt("Invalid argument 4: it should be scalar");
    }
    if (nrhs == 5 && (!mxIsScalar(prhs[4]) || mxIsComplex(prhs[4]))) {
        mexErrMsgTxt("Invalid argument 5: it should be scalar");
    }
    if (nlhs != 2) {
        mexErrMsgTxt("Invalid number of output arguments, they should be 2");
    }

    double sigma = (double)mxGetScalar(prhs[1]);
    int R = (int)mxGetScalar(prhs[2]);
    int d = (int)mxGetScalar(prhs[3]);
    int seed = 0;
    if (nrhs == 5) seed = (int)mxGetScalar(prhs[4]);

    // Create MATLAB delta and mu
    plhs[0] = mxCreateNumericMatrix(d, R, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(d, R, mxDOUBLE_CLASS, mxREAL);
 
    double *delta = (double*)mxGetData(plhs[0]); 
    double *mu = (double*)mxGetPr(plhs[1]);

    GenerateGridParas(delta, mu, d, R, sigma, (KernelType)kernel, seed);
}
