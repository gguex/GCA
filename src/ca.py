import numpy as np
import scipy

# Fonction for eigen decomposition
def sorted_eig(matrix, dim_max=None):
    """
    Compute the eigenvalues and eigenvectors of a real matrix, 
    sorted in descending order.
    
    Parameters:
    matrix (ndarray):   A square matrix for which to compute the eigenvalues 
                        and eigenvectors.
    dim_max (int, optional):    If specified and less than the number of rows 
                                in the matrix minus one, the function will 
                                compute the largest `dim_max` eigenvalues and 
                                corresponding eigenvectors using sparse matrix 
                                methods. Otherwise, it will compute all
                                eigenvalues and eigenvectors.
    Returns:
    tuple: A tuple containing:
        eigen_values (ndarray): The sorted eigenvalues in descending order.
        eigen_vectors (ndarray): The eigenvectors corresponding to the 
        sorted eigenvalues.
    """
    if (dim_max is not None) and dim_max < matrix.shape[0] - 1:
        eigen_values, eigen_vectors = scipy.sparse.linalg.eigs(matrix, dim_max)
    else:
        eigen_values, eigen_vectors = scipy.linalg.eig(matrix)
    sorted_indices = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]

    return np.real(eigen_values), np.real(eigen_vectors)

# Fonction for correspondence analysis
def correspondence_analysis(contingency):
    """
    Perform Correspondence Analysis (CA) on a given contingency table.
    Parameters:
    contingency (array-like): A 2D array representing the contingency table.
    Returns:
    dictionnary: A dictonnary containing the following elements:
        - dim_max (int): The maximum dimension of the CA.
        - eig_val (ndarray): The eigenvalues.
        - row_coord (ndarray): The coordinates of the rows.
        - col_coord (ndarray): The coordinates of the columns.
        - row_contrib (ndarray): The contributions of the rows.
        - col_contrib (ndarray): The contributions of the columns.
        - row_cos2 (ndarray): The cos2 of the rows.
        - col_cos2 (ndarray): The cos2 of the columns.
    """

    contingency = np.array(contingency)
    n_row, n_col = contingency.shape
    dim_max = min(n_row, n_col) - 1

    total = np.sum(contingency)
    f_row = contingency.sum(axis=1)
    f_row = f_row / sum(f_row)
    f_col = contingency.sum(axis=0)
    f_col = f_col / sum(f_col)
    independency = np.outer(f_row, f_col) * total
    normalized_quotient = contingency / independency - 1

    b_mat = (normalized_quotient * f_col) @ normalized_quotient.T
    k_mat = np.outer(np.sqrt(f_row), np.sqrt(f_row)) * b_mat
    eig_val, eig_vec = sorted_eig(k_mat, dim_max)
    eig_val = np.abs(eig_val[:dim_max])
    eig_vec = eig_vec[:, :dim_max]

    row_coord = np.real(np.outer(1 / np.sqrt(f_row), np.sqrt(eig_val)) * eig_vec)
    col_coord = (normalized_quotient.T * f_row) @ row_coord / np.sqrt(eig_val)
    row_contrib = eig_vec ** 2
    col_contrib = np.outer(f_col, 1 / eig_val) * col_coord ** 2
    row_cos2 = row_coord ** 2
    row_cos2 = (row_cos2.T / row_cos2.sum(axis=1)).T
    col_cos2 = col_coord ** 2
    col_cos2 = (col_cos2.T / col_cos2.sum(axis=1)).T

    results = {
        'dim_max': dim_max,
        'eig_val': eig_val,
        'row_coord': row_coord,
        'col_coord': col_coord,
        'row_contrib': row_contrib,
        'col_contrib': col_contrib,
        'row_cos2': row_cos2,
        'col_cos2': col_cos2
    }
    
    return results
    
    