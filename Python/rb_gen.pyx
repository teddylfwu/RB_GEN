cimport numpy as np
import numpy as np
import scipy.sparse
from cython cimport view
from libc.stdlib cimport free

np_size_t = np.uint32 if sizeof(size_t) == 4 else np.uint64

cdef extern from "../generate_grids.h":
    int GenerateGridParas(double *delta, double *mu, size_t d, size_t R, double sigma, int kernel, int seed)

cdef extern from "../common.h":
    cdef enum Mat_type:
        Mat_type_dense = 0
        Mat_type_CSR = 1
    ctypedef struct Mat:  
        Mat_type t "type"
        size_t n
        size_t d
        size_t nnz
        double *val
        size_t *rowIndex
        size_t *colIndex

cdef extern from "../feature_matrix.h":
    int ComputeTrainFeatureMatrix(size_t R, Mat X, double *delta, double *mu, size_t *offset, int **codeTable, Mat *Phi)
    int ComputeTestFeatureMatrix(size_t R, Mat X, double *delta, double *mu, size_t *offset, int *codeTable, Mat *Phi)

def rb_grid(kernel, sigma, R, d, seed=0):
    """
    Generates the random orthogonal grid used for generating the random binding
    map.

    This function returns the widths of the grids, which are generated from the
    distribution indicated by kernel with standard deviation sigma, and the
    bias of the grids, which are generated as the uniform distribution [0,1]
    times the widths.

    Parameters
    ----------
    kernel: str ['Gaussian' | 'Laplace' | 'ProdLaplace']
        Distribution used to generate the grid's widths. One of the next
        options: 

            "Gaussian": Gaussian distribution

            "Laplace": multivariate t-Student

            "ProdLaplace": t-Student

    sigma: real
        Standard deviation for the distribution that generates the widths.

    R: int
        Number of random grids generated.

    d: int
        Dimension of the grids

    seed: int, optional
        Initial seed of the random generator.

    Returns
    -------
    delta : array
        Array of R x d for the widths

    mu : array
        Array of R x d for the bias

    Examples
    --------
    >>> import rb_gen
    >>> delta, mu = rb_gen.rb_grid("Gaussian", .5, 2, 4)
    >>> delta
    array([[-0.92978526,  0.7270068 ,  0.41912077, -1.3342806 ],
           [ 0.27838466,  0.81395224,  0.3379512 , -2.93759582]])
    >>> mu
    array([[-0.25827083,  0.40273992,  0.20008702, -0.83909028],
           [ 0.1015504 ,  0.41788382,  0.32180717, -2.6914108 ]])
    """

    Kernels = {"Gaussian": 0, "Laplace": 1, "ProdLaplace": 2}
    if kernel not in Kernels.keys():
        raise ValueError("Not valid value for `kernel'")

    if sigma < 0:
        raise ValueError("Not valid value for `sigma': it should be positive")

    if R < 0 or not int(R) == R:
        raise ValueError("Not valid value for `R': it should a positive integer.")

    if d < 0 or not int(d) == d:
        raise ValueError("Not valid value for `d': it should a positive integer.")

    if not int(seed) == seed:
        raise ValueError("Not valid value for `seed': it should an integer.")

    # Create delta and mu in F order
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] delta = np.zeros((R,d), dtype=np.double, order='c')
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] mu = np.zeros((R,d), dtype=np.double, order='c')

    # Call the main function
    GenerateGridParas(&delta[0,0], &mu[0,0], d, R, sigma, Kernels[kernel], seed)

    return delta, mu

def rb_train(A, sigma=None, R=None, kernel=None, seed=0, mu=None, delta=None, return_phi=False):
    """
    Compute the coordinates of the nonempty bins after coarsening the data on
    the given random grids specified by mu and delta. Alternatively, it can
    be given the information to generate the grids. See kernel, sigma and R.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix or numpy.ndarray
         Input matrix. The rows are mapped into coordinates on the random grids.

    kernel: str ['Gaussian' | 'Laplace' | 'ProdLaplace'], optional
        Distribution used to generate the grid's widths. One of the next
        options: 

            "Gaussian": Gaussian distribution

            "Laplace": multivariate t-Student

            "ProdLaplace": t-Student

        Default values is 'Gaussian'.

    sigma: real, optional
        Standard deviation for the distribution that generates the widths.

    R: int, optional
        Number of random grids generated.

    seed: int, optional
        Initial seed of the random generator.

    delta: numpy.ndarray
        Array of widths. It has dimensions R x d, where d is the number of
        columns in A and R is the number of random grids.
    
    mu: numpy.ndarray
        Array of bias. It has dimensions R x d, where d is the number of
        columns in A and R is the number of random grids.

    return_phi: bool
        Whether to return the transformation of A.

        The default is False

    Returns
    -------
    offset : array
        The range offset[i+1]:offset[i], indicates the columns of matrix coor
        corresponding to the nonempty bins for gird i-th.

    coor : numpy.ndarra
        Matrix whose columns are the coordinates of the nonempty bins.

    delta : array
        Array of R x d for the widths. Returned if it is not passed as input

    mu : array
        Array of R x d for the bias. Returned if it is not passed as input

    phi : scipy.sparse.csr_matrix
        Feature matrix of A. Every row of A is replaced by the R indices of
        the coordinates of the random grids. Returned if return_phi is True.

    Examples
    --------
    >>> import rb_gen
    >>> import numpy as np
    >>> A = np.random.random((4,3))
    >>> offset, coor, delta, mu = rb_gen.rb_train(A, sigma=.5, R=2)
    >>> offset
    array([0, 4, 8], dtype=uint64)
    >>> coor
    array([[-2, -1,  1],
           [-1, -1,  1],
           [-2,  0,  1],
           [-1, -1,  0],
           [-2,  1,  0],
           [-1,  0,  0],
           [-2,  2,  0],
           [-1, -1, -1]], dtype=int32)
    >>> delta
    array([[-0.92978526,  0.7270068 ,  0.41912077],
           [-1.3342806 ,  0.27838466,  0.81395224]])
    >>> mu
    array([[-0.31168518,  0.55850814,  0.11642115],
           [-0.73915137,  0.13290002,  0.51187089]])

    The above is equivalent to:

    >>> delta, mu = rb_gen.rb_grid("Gaussian", .5, 3, A.shape[1])
    >>> offset, coor = rb_gen.rb_train(A, delta=delta, mu=mu)
    """

    if (delta is None) != (mu is None):
        raise ValueError("Either delta and mu are not None or both are not None")

    if delta is not None and  (sigma is not None or R is not None or kernel is not None):
        raise ValueError("Either you set delta and mu or you set sigma, R and kernel")

    delta_mu_given = True
    if delta is None:
        if kernel is None: kernel = "Gaussian"
        delta, mu = rb_grid(kernel, sigma, R, A.shape[1], seed)
        delta_mu_given = False

    if delta.shape[1] != A.shape[1]:
        raise ValueError("Invalid shape for delta. It should have as many columns as A")

    if mu.shape[1] != A.shape[1]:
        raise ValueError("Invalid shape for mu. It should have as many columns as A")

    if mu.shape[0] != delta.shape[0]:
        raise ValueError("Invalid shape for mu. It should have the same shape as delta")

    # Translate A into Mat

    cdef Mat X
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] Ac
    cdef np.ndarray[size_t, ndim=1] X_indices, X_indptr
    cdef np.ndarray[np.double_t, ndim=1] X_data
    if isinstance(A, np.ndarray):
        # Input matrix is dense
        X.t = Mat_type_dense
        X.n = A.shape[0]
        X.d = A.shape[1]
        X.nnz = X.n * X.d
        # Force matrix in C mode (row-major)
        Ac = np.ascontiguousarray(A.astype(np.double, copy=False, order='C'))
        X.val = &Ac[0,0]
        X.rowIndex = NULL
        X.colIndex = NULL
    elif scipy.sparse.issparse(A):
        # Input matrix is sparse
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()
        X.t = Mat_type_CSR
        X.n = A.shape[0]
        X.d = A.shape[1]
        X.nnz = A.nnz
        X_data = A.data.astype(np.double)
        X.val = &X_data[0]
        X_indptr = A.indptr.astype(np_size_t)
        X.rowIndex = &X_indptr[0]
        X_indices = A.indices.astype(np_size_t)
        X.colIndex = &X_indices[0]

    # Force contiguous arrays in delta and mu

    cdef np.ndarray[np.double_t, ndim=2, mode='c'] deltac = delta.astype(np.double, copy=False, order='c')
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] muc = mu.astype(np.double, copy=False, order='c')

    # Create offset

    cdef np.ndarray[size_t, ndim=1] offset = np.zeros(delta.shape[0]+1, dtype=np_size_t)

    # Create CSR arrays for phi if needed

    cdef Mat Phi
    cdef Mat *phiptr = NULL
    cdef np.ndarray[size_t, ndim=1] Phi_indices, Phi_indptr
    cdef np.ndarray[np.double_t, ndim=1] Phi_data
    if return_phi:
        phiptr = &Phi
        Phi.t = Mat_type_CSR
        Phi.n = X.n
        Phi.nnz = X.n*delta.shape[0]
        Phi_data = np.zeros(Phi.nnz, dtype=np.double)
        Phi_indices = np.zeros(Phi.nnz, dtype=np_size_t)
        Phi_indptr = np.zeros(A.shape[0]+1, dtype=np_size_t)
        Phi.val = &Phi_data[0]
        Phi.rowIndex = &Phi_indptr[0]
        Phi.colIndex = &Phi_indices[0]

    # Calling the main function

    cdef int *coorc
    ComputeTrainFeatureMatrix(delta.shape[0], X, &deltac[0,0], &muc[0,0],
        &offset[0], &coorc, phiptr)

    # Create coor

    cdef view.array coor_view = view.array(
            shape=(offset[delta.shape[0]],A.shape[1]), itemsize=sizeof(int),
            format='i', mode='c', allocate_buffer=False)
    coor_view.data = <char *>coorc
    coor_view.callback_free_data = free
    coor = np.asarray(coor_view)

    # Create phi

    if return_phi:
        phi = scipy.sparse.csr_matrix((Phi_data, Phi_indices, Phi_indptr),
                shape=(A.shape[0], offset[delta.shape[0]]))

    # Return

    if return_phi:
        if delta_mu_given:
            return offset, coor, phi
        else:
            return offset, coor, delta, mu, phi
    else:
        if delta_mu_given:
            return offset, coor
        else:
            return offset, coor, delta, mu


def rb_test(A, offset, coor, delta, mu):
    """
    Compute the coordinates of the nonempty bins after coarsening the data on
    the given random grids specified by mu and delta.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix or numpy.ndarray
         Input matrix. The rows are mapped into coordinates on the random grids.

    offset : array
        The range offset[i+1]:offset[i], indicates the columns of matrix coor
        corresponding to the nonempty bins for gird i-th.

    coor : numpy.ndarra
        Matrix whose columns are the coordinates of the nonempty bins.

    
    delta: numpy.ndarray
        Array of widths. It has dimensions R x d, where d is the number of
        columns in A and R is the number of random grids.
    
    mu: numpy.ndarray
        Array of bias. It has dimensions R x d, where d is the number of
        columns in A and R is the number of random grids.

    Returns
    -------
    phi : scipy.sparse.csr_matrix
        Feature matrix of A. Every row of A is replaced by the R indices of
        the coordinates of the random grids. 

    Examples
    --------
    >>> import rb_gen
    >>> import numpy as np
    >>> A = np.random.random((4,3))
    >>> offset, coor, delta, mu = rb_gen.rb_train(A, sigma=.5, R=2)
    >>> B = np.random.random((10,3))
    >>> phi = rb_gen.rb_test(B, offset, coor, delta, mu)
    """

    if offset[offset.shape[0]-1] != coor.shape[0]:
        raise ValueError("Invalid shape for coor. It should have offset[-1] rows.")

    if delta.shape[1] != A.shape[1]:
        raise ValueError("Invalid shape for delta. It should have as many columns as A")

    if mu.shape[1] != A.shape[1]:
        raise ValueError("Invalid shape for mu. It should have as many columns as A")

    if mu.shape[0] != delta.shape[0]:
        raise ValueError("Invalid shape for mu. It should have the same shape as delta")

    if delta.shape[0] != offset.shape[0]-1:
        raise ValueError("Invalid shape for delta. It should have as many rows as len(offset)-1.")

    # Translate A into Mat

    cdef Mat X
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] Ac
    cdef np.ndarray[size_t, ndim=1] X_indices, X_indptr
    cdef np.ndarray[np.double_t, ndim=1] X_data
    if isinstance(A, np.ndarray):
        # Input matrix is dense
        X.t = Mat_type_dense
        X.n = A.shape[0]
        X.d = A.shape[1]
        X.nnz = X.n * X.d
        # Force matrix in C mode (row-major)
        Ac = np.ascontiguousarray(A.astype(np.double, copy=False, order='C'))
        X.val = &Ac[0,0]
        X.rowIndex = NULL
        X.colIndex = NULL
    elif scipy.sparse.issparse(A):
        # Input matrix is sparse
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()
        X.t = Mat_type_CSR
        X.n = A.shape[0]
        X.d = A.shape[1]
        X.nnz = A.nnz
        X_data = A.data.astype(np.double)
        X.val = &X_data[0]
        X_indptr = A.indptr.astype(np_size_t)
        X.rowIndex = &X_indptr[0]
        X_indices = A.indices.astype(np_size_t)
        X.colIndex = &X_indices[0]

    # Force contiguous arrays in offset, coor, delta and mu

    cdef np.ndarray[size_t, ndim=1] offsetc = offset.astype(np_size_t, copy=False)
    cdef np.ndarray[int, ndim=2, mode='c'] coorc = coor.astype(np.intc, copy=False, order='c')
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] deltac = delta.astype(np.double, copy=False, order='c')
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] muc = mu.astype(np.double, copy=False, order='c')

    # Create CSR arrays for phi

    cdef Mat Phi
    cdef np.ndarray[size_t, ndim=1] Phi_indices, Phi_indptr
    cdef np.ndarray[np.double_t, ndim=1] Phi_data
    Phi.t = Mat_type_CSR
    Phi.n = X.n
    Phi.nnz = X.n*delta.shape[0]
    Phi_data = np.zeros(Phi.nnz, dtype=np.double)
    Phi_indices = np.zeros(Phi.nnz, dtype=np_size_t)
    Phi_indptr = np.zeros(A.shape[0]+1, dtype=np_size_t)
    Phi.val = &Phi_data[0]
    Phi.rowIndex = &Phi_indptr[0]
    Phi.colIndex = &Phi_indices[0]

    # Calling the main function

    ComputeTestFeatureMatrix(delta.shape[0], X, &deltac[0,0], &muc[0,0],
        &offsetc[0], &coorc[0,0], &Phi)

    # Create phi

    phi = scipy.sparse.csr_matrix((Phi_data, Phi_indices, Phi_indptr),
                shape=(A.shape[0], coor.shape[0]))

    # Return

    return phi
