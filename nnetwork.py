import numpy as np
from costfunction import sigmoid

class NeuralNet:
    def __init__(self, nNodesInLayer, lambd):
        assert type(nNodesInLayer) is np.ndarray
        assert len(nNodesInLayer) > 1
        self.nLayer = len( nNodesInLayer)
        self.nNodes = np.sum( nNodesInLayer )
        self.NodesInLayer = nNodesInLayer
        self.theta = []
        self.al = []
        self.dl = []
        self.deriv = []
        self.regParam = lambd
        for ii in range( self.nLayer - 1 ):
            nIn  = self.NodesInLayer[ ii ]
            nOut = self.NodesInLayer[ ii + 1 ]
            self.theta.append( np.random.uniform( size=(nOut, nIn+1) ) )
        print 'Neural Network created!'
        print self.nLayer, 'layers in total (including In- and Output-layer)'
        print len( self.theta ), 'parameter matrices'


    def get_nodes_in_layer(self, n):
        return self.NodesInLayer[ n ]

    def get_theta(self, n):
        return self.theta[ n ]

    def get_al(self, n):
        return self.al[n]

    def get_dl(self, n):
        return self.dl[n]

    def get_derivatives(self, n):
        return self.deriv[n]

    def forward_propagate(self, xIn):
        xTMP = xIn
        self.al = []
        for theta in self.theta:
            xTMP = np.vstack( ( np.ones( np.shape(xIn)[1]), xTMP ) )
            self.al.append( xTMP )
            xTMP = theta.dot( xTMP )
            xTMP = sigmoid( xTMP )
        self.al.append( xTMP )
        return None

    def back_propagate(self, xOut):
        xTMP = xOut
        self.dl = []
        self.dl.append( xTMP )
        for ii in range( len(self.theta) ):
            idx = len(self.theta) - ii - 1
            thetaTMP = self.theta[idx]
            thetaTMP = np.delete(thetaTMP, 0, 1)
            xTMP  = thetaTMP.T.dot( xTMP )
            alTMP = np.delete( self.al[idx], 0, 0)
            xTMP = xTMP * alTMP * (1. - alTMP )
            self.dl.append( xTMP )
        self.dl = self.dl[ ::-1 ]
        return None

    def calculate_derivatives(self):
        self.deri = []
        for ii in range( len(self.theta) ):
            ai = self.al[ ii ]
            di = self.dl[ ii + 1 ]
            yDim = np.shape(ai)[0]
            xDim = np.shape(di)[0]
            sumTMP = np.zeros( (xDim,yDim) )
            for jj in range( np.shape(ai)[1] ):
                sumTMP += np.outer( di.T[jj], ai.T[jj] )
            thetaTMP = self.theta[ii]   
            thetaTMP.T[0] = 0
            self.deriv.append( 1. * sumTMP/np.shape(ai)[1] 
                               + self.regParam * thetaTMP )
        return None 

    def update_theta(self, epsilon):
        for ii in range( len(self.theta) ):
            self.theta[ii] = ( 1. - epsilon * self.deriv[ii] ) * self.theta[ii]
        return None


###Check dimensions for easy example
###Check derivatives

nLayers = np.array( [3,2,1] )
NN = NeuralNet( nLayers,1 )
xTest = np.array( [ [1., 5., 6., 4.], [2., 10., 2., 4.], [1., 5., 3., 4.] ] )
NN.forward_propagate(xTest)
xOut = NN.get_al( len(nLayers) - 1 )
NN.back_propagate( xOut )
NN.calculate_derivatives()

print NN.get_theta(0)
NN.update_theta( 0.1 )
print NN.get_theta(0)

print ''
print NN.get_derivatives(0)

