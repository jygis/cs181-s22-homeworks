#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40

    if part == "a" and not is_years:
        xx = xx/20

    m = len(xx) #first dimension of the output array

    if part == 'a':
        ret = np.ones((m,6))
        for i in range(1,6):
            ret[:,i] = xx**i

    if part == 'b':
        ret = np.ones((m,12))
        mus = [x for x in range(1960,2011,5)]
        for i in range(len(mus)):
            ret[:,i] = np.exp(-(xx - mus[i]) ** 2 / 25)

    if part == 'c':
        ret = np.ones((m,6))
        for i in range(1,6):
            ret[:,i] = np.cos(xx/i)

    if part == 'd':
        ret = np.ones((m,26))
        for i in range(1,26):
            ret[:,i] = np.cos(xx/i)


    return ret

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

plt.figure(figsize=(10, 10), dpi=200)
p = ['a','b','c','d']
for j in range(4):
    # DO NOT CHANGE grid_years!!!!!
    grid_years = np.linspace(1960, 2005, 200)
    grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
    grid_X_basis = make_basis(grid_X.T[:,1], part = p[j])

    #Use Training Data to Find Weights
    X_train = X[:,1]
    X_train_basis = make_basis(X_train, part = p[j])
    w = find_weights(X_train_basis, Y)

    #Use weights on the grid_X data
    grid_Yhat  = np.dot(grid_X_basis, w)

    # TODO: plot and report sum of squared error for each basis
    SSE = sum((Y - np.dot(X_train_basis, w))**2)

    # Plot the data and the regression line.
    plt.subplot(2, 2, j+1)
    plt.plot(X[:,1], Y, 'o', label = 'Original Data')
    plt.plot(grid_years, grid_Yhat, label = f'Regression with Basis {p[j]}')
    plt.text(1960,54,f'SSE is {SSE:.4f}')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.legend()

sunspot_counts_use = sunspot_counts[:13]
X = np.vstack((np.ones(sunspot_counts_use.shape), sunspot_counts_use)).T
Y = Y[:13]

plt.figure(figsize=(10, 10), dpi=200)
p = ['a','c','d']
for j in range(3):
    grid_sun = np.linspace(min(sunspot_counts_use), max(sunspot_counts_use), 200)
    grid_X = np.vstack((np.ones(grid_sun.shape), grid_sun))
    grid_X_basis = make_basis(grid_X.T[:,1], part = p[j], is_years=False)

    #Use Training Data to Find Weights
    X_train = X[:,1]
    X_train_basis = make_basis(X_train, part = p[j], is_years=False)
    w = find_weights(X_train_basis, Y)

    #Use weights on the grid_X data
    grid_Yhat  = np.dot(grid_X_basis, w)

    # TODO: plot and report sum of squared error for each basis
    SSE = sum((Y - np.dot(X_train_basis, w))**2)

    # Plot the data and the regression line.
    plt.subplot(2, 2, j+1)
    plt.plot(X[:,1], Y, 'o', label = 'Original Data')
    plt.plot(grid_sun, grid_Yhat, label = f'Regression with Basis {p[j]}')
    plt.title(f'SSE is {SSE:.4f}', ha = 'center')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.legend()
