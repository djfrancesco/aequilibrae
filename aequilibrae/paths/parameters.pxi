import numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int64
ctypedef np.int64_t ITYPE_t

# EPS is the precision of DTYPE
cdef DTYPE_t DTYPE_EPS = 1E-15

# NULL_IDX is the index used in predecessor matrices to store a non-path
cdef ITYPE_t NULL_IDX = 18446744073709551615

VERSION = 0.7
MINOR_VRSN = 7
binary_version = "0.7.7"
release_name = "Queluz"
