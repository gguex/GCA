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
    
    output_contingency = contingency.copy()
    
    if S_r is not None:
        f_r = contingency.sum(axis=1) / contingency.sum()
        psi_r = np.ones(contingency.shape[0])
        convergence = False
        while not convergence:
            psi_r_old = psi_r
            phi_r = 1 / (psi_r * S_r).sum(axis=1)
            psi_r = f_r / (f_r * phi_r * S_r.T).sum(axis=1)
            convergence = np.abs(psi_r - psi_r_old).sum() < 1e-6
        markovcm_r = np.outer(phi_r, psi_r) * S_r
        output_contingency = markovcm_r.T @ output_contingency
    
    if S_c is not None:
        f_c = contingency.sum(axis=0) / contingency.sum()
        psi_c = np.ones(contingency.shape[1])
        convergence = False
        while not convergence:
            psi_c_old = psi_c
            phi_c = 1 / (psi_c * S_c).sum(axis=1)
            psi_c = f_c / (f_c * phi_c * S_c.T).sum(axis=1)
            convergence = np.abs(psi_c - psi_c_old).sum() < 1e-6
        markovcm_c = np.outer(phi_c, psi_c) * S_c
        output_contingency = output_contingency @ markovcm_c
    
    return output_contingency