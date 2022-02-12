#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

y = np.array(data)[:,1]
x = np.array(data)[:,0]

def compute_loss(tau):
    # TODO
    loss = 0
    for n in range(len(data)):
        sigma_i = 0
        for i in range(len(data)):
            if i != n:
                d = np.matmul(np.matrix(np.transpose(x[i] - x[n])), np.matrix(x[i] - x[n]))
                sigma_i += np.exp(-d/tau) * y[i]
        loss += (y[n] - sigma_i) ** 2
    return loss.item()

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))
