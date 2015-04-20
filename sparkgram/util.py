"""
Utility functions for Sparkgram
========================
"""
from scipy.sparse import csr_matrix
import numpy as np, scipy
from functools import partial


def make_csr_matrix_index(max_index) : 
    return partial(make_csr_matrix, max_index = max_index)

def make_csr_matrix(features, max_index) : 

    indices = np.array([p[0] for p in features], dtype = np.int64)
    values  = np.array([p[1] for p in features], dtype = np.float64)
    
    assert(np.alltrue(indices >= 0))
    assert(np.alltrue(values >= 0))
    
    if len(indices) > 0 : 
        return csr_matrix((values, (np.zeros(len(indices)), indices)), shape=(1,max_index+1))
     
def calculate_column_stat(rdd, op = 'mean') : 
    """Given an RDD of sparse feature vectors, calculate various quantities
    `rdd` should be an rdd of `scipy.sparse.csr_matrix` 

    `op` can be 'mean' or 'norm' 
    """ 
    n_items = rdd.count()

    if op == 'mean':
        res = rdd.reduce(np.add)
    if op == 'norm': 
        res = rdd.map(square_csr).reduce(np.add)
        res.data = np.sqrt(res.data)

    if res.shape[1]*8 < (res.data.nbytes + res.indptr.nbytes + res.indices.nbytes) : 
        res = res.toarray().squeeze()

    return res

def square_csr(vec) : 
    """Square the individual elements of a csr matrix"""
    vec.data *= vec.data
    return vec


def add_arrays(arr1, arr2) : 
    """Add the two sparse arrays but return a dense array if it saves memory"""
    res = arr1 + arr2
    if type(res) is scipy.sparse.csr.csr_matrix: 
        if res.shape[1]*8 < (res.data.nbytes + res.indptr.nbytes + res.indices.nbytes) : 
            return res.toarray()
    return res
