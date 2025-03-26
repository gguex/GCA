# ---------------------------------------------------
# Toy example for the generalized CA
# ---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from src.ca import correspondence_analysis
from src.gca import contingency_with_sim

# Create a contingency table
n_r = 7
n_c = 4
cont_table = np.random.randint(0, 11, (n_r, n_c))

# Create the grouping matrix
Z_crisp = np.array([[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1]])
n_g = 3

# Create the crisp similarity 
S_crisp = Z_crisp @ Z_crisp.T

# Perform the CA
res = correspondence_analysis(cont_table)

# Perform the grouped CA
cont_table_gpd = Z_crisp.T @ cont_table
res_gpd = correspondence_analysis(cont_table_gpd)

# Perform the CA with similarity matrices
cont_table_sim = contingency_with_sim(cont_table, S_r=S_crisp)
res_sim = correspondence_analysis(cont_table_sim)

# Plot the CA results
fig, ax = plt.subplots()
ax.scatter(res['row_coord'][:, 0], res['row_coord'][:, 1], color='blue')
ax.scatter(res['col_coord'][:, 0], res['col_coord'][:, 1], color='red')
for i in range(n_r):
    ax.text(res['row_coord'][i, 0], res['row_coord'][i, 1], f'Row {i+1}')
for j in range(n_c):
    ax.text(res['col_coord'][j, 0], res['col_coord'][j, 1], f'Col {j+1}')
plt.show()

# Plot the grouped CA results
fig, ax = plt.subplots()
ax.scatter(res_gpd['row_coord'][:, 0], res_gpd['row_coord'][:, 1], color='blue')
ax.scatter(res_gpd['col_coord'][:, 0], res_gpd['col_coord'][:, 1], color='red')
for i in range(n_g):
    ax.text(res_gpd['row_coord'][i, 0], res_gpd['row_coord'][i, 1], f'Gp {i+1}')
for j in range(n_c):
    ax.text(res_gpd['col_coord'][j, 0], res_gpd['col_coord'][j, 1], f'Col {j+1}')
plt.show()

# Plot the CA with similarity matrices results
fig, ax = plt.subplots()
ax.scatter(res_sim['row_coord'][:, 0], res_sim['row_coord'][:, 1], color='blue')
ax.scatter(res_sim['col_coord'][:, 0], res_sim['col_coord'][:, 1], color='red')
for i in range(n_r):
    ax.text(res_sim['row_coord'][i, 0], res_sim['row_coord'][i, 1], f'Row {i+1}')
for j in range(n_c):
    ax.text(res_sim['col_coord'][j, 0], res_sim['col_coord'][j, 1], f'Col {j+1}')
plt.show()


