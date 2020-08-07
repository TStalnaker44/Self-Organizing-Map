"""
File: som.py
Author: Trevor Stalnaker
Assignment 2
Date: 28 September 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Generates animation of the learning process
"""
def animate(data, som, maxIter, alpha_0, d_0):
    som.learn(data, maxIter, alpha_0, d_0, True)
    plt.show()
    
"""
Plots the before and after results of the Kohonen Net's learning
Can plot in 2 and 3 dimensions
"""
def plotAll(data, som, t, alpha0, d0, title):
    plot(data, som, alpha0, d0, 0, title)
    som.learn(data, t, alpha0, d0)
    plot(data, som, alpha0, d0, t, title)

"""
Responsible for plotting the SOM and the test data
Works with both 2 and 3 dimensional plots
""" 
def graph(data, som, alpha0, d0, t, title):
    if(som.n == 3):
        plt3d = plt.figure().gca(projection='3d')
        plt3d.scatter(data[:,0], data[:,1], data[:,2], s=0.2)
        drawLattice(som, plt3d)
    else:
        plt.scatter(data[:,0], data[:,1], s=.2)
        plt.gca().set_aspect('equal')
        drawLattice(som)
    xs,ys,zs = [],[],[]
    for j in range(som.m):
        for k in range(som.m):
            xs.append(som.weights[j,k][0])
            ys.append(som.weights[j,k][1])
            if(som.n == 3):
                zs.append(som.weights[j,k][2])
    plt.title(title + "\nM=" + str(som.m) + ", alpha0=" + str(alpha0) +
              ", d0=" + str(d0) + ", T=" + str(t))
    if (som.n == 3):
        plt3d.plot(xs,ys,zs,'ro')
    else:
        plt.plot(xs,ys,'ro')

"""
Plot and show the SOM and the data
"""
def plot(data, som, alpha0, d0, t, title):
    graph(data, som, alpha0, d0, t, title)
    plt.show()
    
"""
Draws the lattice connecting all of the neurons in a SOM
"""
def drawLattice(som, plotter=plt):
    hasPrev = False
    for k in range(som.m):
        for j in range(som.m):
            h1_comps, h2_comps, v1_comps, v2_comps = [],[],[],[]
            neuron_h = som.weights[j,k]
            neuron_v = som.weights[k,j]
            for n in range(som.n):
                h1_comps.append(neuron_h[n])
                v1_comps.append(neuron_v[n])
            if (hasPrev):
                for n in range(som.n):
                    h2_comps.append(prevNeuron_h[n])
                    v2_comps.append(prevNeuron_v[n])
                plotLine(h1_comps, h2_comps, plotter)
                plotLine(v1_comps, v2_comps, plotter)
            prevNeuron_h, prevNeuron_v = neuron_h, neuron_v
            hasPrev = True
        hasPrev = False

"""
Plots the connection between two points
"""
def plotLine(pointOne, pointTwo, plotter):
    if (len(pointOne) == 2):
        plotter.plot([pointOne[0],pointTwo[0]],
                 [pointOne[1],pointTwo[1]],'b')
    if (len(pointOne) == 3):
        plotter.plot([pointOne[0],pointTwo[0]],
                 [pointOne[1],pointTwo[1]],
                 [pointOne[2],pointTwo[2]],'b')
        
"""
A SOM object
"""
class SOM():
    
    """
    Param m - length/width for a square grid
    Param n - dimensionality of the weights
    """
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.weights = (np.random.random((m,m,n))/10)+0.45

    """
    Method responsible for updating the weights of the SOM
    """
    def learn(self, data, maxIter, alpha_0, d_0, animate=False):
        
        #Iterate t from 0 to T
        for t in range(maxIter):
                
            #Compute current neighborhood radius d and learning rate alpha
            d_t = int(np.ceil(d_0 * (1 - (t / maxIter))))
            alpha_t = alpha_0 * (1 - (t / maxIter))
            
            #Pick an input e from the training set at random
            e = data[np.random.random_integers(len(data) - 1)]
            #print(e)
            
            #Find the winning unit whose weights are closest to this e
            jwin, kwin = self.findWinner(e)

            #Loop over the neighbors of this winner, adjusting their weights
            neighbors = self.getNeighbors(jwin, kwin, d_t)
            for neighbor in neighbors:
                 u = self.weights[neighbor[0],neighbor[1]]
                 u += (alpha_t * (e - u))
                 self.weights[neighbor[0],neighbor[1]] = u

            if (animate == True and t % 100 == 0):
                plt.clf()
                graph(data, som, alpha_0, d_0, t, "Animation")
                plt.pause(0.05)
                     
    """
    Find the winner, given a vector e
    """
    def findWinner(self, e):
        minDist = float('Inf')
        jwin = 0
        kwin = 0
        for j in range(self.m):
            for k in range(self.m):
                dist = np.sum((e - self.weights[j,k])**2)
                if (dist < minDist):
                    minDist = dist
                    jwin, kwin = j, k
        return jwin, kwin

    """
    Find the neighbors of a neuron given its j and k position and a neighborhood size
    """
    def getNeighbors(self, j, k, d):
        j_min = j - d
        if (j_min < 0): j_min = 0
        j_max = j + d
        if (j_max > self.m - 1): j_max = self.m - 1
        k_min = k - d
        if (k_min < 0): k_min = 0
        k_max = k + d
        if (k_max > self.m - 1): k_max = self.m - 1

        indexLyst = []
        for j_comp in range(j_min, j_max+1):
            for k_comp in range(k_min, k_max+1):
                indexLyst.append((j_comp, k_comp))        
        return indexLyst

#Parts 1-4 Learning with a Square and Building the SOM
print("Close pop up windows to advance")
print("Plotting parts one through four...")
data = np.random.random((5000, 2))
som = SOM(8, 2)
plotAll(data, som, 4000, 0.2, 4,"Square Training Set")

#Part 5 Learning with a Ring
print("Plotting part five...")
r = (data[:,0]-0.5)**2 + (data[:,1]-0.5)**2
subset = data[np.logical_and(r<0.2,r>0.1)]
som = SOM(8,2)
plotAll(subset, som, 4000, 0.2, 4, "Ring Training Set")

#Extra Credit
print("Extra Credit...")

#Learn with a Cross
print("Learning with a cross...")
x, y = data[:,0], data[:,1]
crossSet = data[np.logical_or(np.logical_and(x>.4,x<.6),np.logical_and(y>.4,y<.6))]
som = SOM(8,2)
plotAll(crossSet, som, 4000, 0.2, 4, "Cross Training Set")

#Learn with an X
#diagOne = y / x
#diagTwo = (y-1)/-x
#xSet = data[np.logical_or(np.logical_and(diagOne>.9,diagOne<1),np.logical_and(diagTwo>.9,diagTwo<1))]
#plotAll(xSet, som, 4000, 0.2, 4, "X Training Set")

#Learn with a Semi-Ring
#semiRingSet = data[np.logical_and(np.logical_and(r<0.2,r>0.1),x<0.5)]
#plotAll(semiRingSet, som, 4000, 0.2, 4, "Semi-Ring Training Set")

#Three Dimensions
print("Three dimensions...")
som3d = SOM(8,3)
data3d = np.random.random((5000,3))
plotAll(data3d, som3d, 4000, 0.2, 4, "3D Cube Training Set")

#Animations
print("Generating animations... (Do not close window until animation has stopped)")
som = SOM(8, 2)
animate(data, som, 4000, 0.2, 4)
animate(subset, som, 4000, 0.2, 4)
animate(crossSet, som, 4000, 0.2, 4)

#Print to the user that all tests have been completed
print("DONE")
