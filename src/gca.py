import numpy as np

# Fonction for computation of the new contingency table
# with similariy matrices
def contingency_with_sim(contingency, S_r=None, S_c=None):
    """
    Compute the new contingency table with the given similarity matrices.
    Parameters:
    contingency (array-like): A 2D array representing the contingency table.
    S_r (array-like): A 2D array representing the similarity matrix for rows
    S_c (array-like): A 2D array representing the similarity matrix for columns
    Returns:
    ndarray: The new contingency table.
    """
    
    contingency = np.array(contingency)
    Z = np.array(Z)
    
    return Z.T @ contingency