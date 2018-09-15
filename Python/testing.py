import rb_gen
import numpy as np
import scipy.sparse

A = np.random.random((4,3))
offset, coor, delta, mu, phi = rb_gen.rb_train(A, sigma=.5, R=2, return_phi=True)
phi0 = rb_gen.rb_test(scipy.sparse.coo_matrix(A), offset, coor, delta, mu)
assert(abs(phi - phi0).max() < np.finfo(np.float64).eps*10)
