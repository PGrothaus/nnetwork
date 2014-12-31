import numpy as np

def sigmoid(z):
    return 1. / ( 1. + np.exp( -z ) )

def cost( xOut, yData, regParam, thetaList ):
    m = np.shape( xOut )[1]
    sumTheta = 0.
    for theta in thetaList:
        theta = np.delete( theta, 0, 1 )
        sumTheta += regParam * np.sum( theta * theta )
    sumCost = - yData * np.log( xOut )
    sumCost -= ( 1. - yData ) * np.log( 1. - xOut )
    sumCost = np.sum( sumCost )
    return 1./m * (sumCost + 1./2 * sumTheta )
