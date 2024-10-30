import numpy as np


def initialization(SearchAgents_no, dim, ub, lb):
    ubarr = np.array([ub])
    lbarr = np.array([lb])
    Boundary_no = ubarr.shape[0]  # numnber of boundaries (shap[0]vi mang 1 chieu)
    # % If the boundaries of all variables are equal and user enter a signle
    # % number for both ub and lb
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim)*(ub-lb)+lb
    # % If each variable has a different lb and ub
    if Boundary_no > 1:
        for i in range(dim):
            ub_i = ubarr[i]
            lb_i = lbarr[i]
            Positions[: , i] = np.random.rand(SearchAgents_no, dim)*(ub_i-lb_i)+lb_i
    return Positions
