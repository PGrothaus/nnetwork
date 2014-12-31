import numpy as np
from costfunction import sigmoid, cost

class NeuralNet:
    def __init__(self, nNodesInLayer, lambd):
        assert type(nNodesInLayer) is np.ndarray
        assert len(nNodesInLayer) > 1
        self.nLayers = len( nNodesInLayer)
        self.nNodes = np.sum( nNodesInLayer )
        self.NodesInLayer = nNodesInLayer
        self.theta = []
        self.al = []
        self.dl = []
        self.deriv = []
        self.regParam = lambd
        for ii in range( self.nLayers - 1 ):
            nIn  = self.NodesInLayer[ ii ]
            nOut = self.NodesInLayer[ ii + 1 ]
            self.theta.append( np.random.uniform( size=(nOut, nIn+1) ) )
            #self.theta.append( np.ones( (nOut, nIn+1) ) )
        print ''
        print 'NEURAL NETWORK created!'
        print self.nLayers, 'layers in total (including In- and Output-layer)'
        print len( self.theta ), 'parameter matrices'
        print ''


    def get_nodes_in_layer(self, n):
        return self.NodesInLayer[ n ]

    def get_theta(self, n):
        assert n < self.nLayers - 1,  'n too large. Not so many '\
                                        'parameter matrices'
        return self.theta[ n ]

    def get_al(self, n):
        return self.al[n]

    def get_dl(self, n):
        return self.dl[n]

    def get_derivatives(self, n):
        assert n < self.nLayers - 1,  'n too large. Not so many layers'
        return self.deriv[n]

    def forward_propagate(self, xIn):
        #Need to add bias term to xIN -> vstack
        assert np.shape(xIn)[0] + 1 == np.shape(self.theta[0])[1]
        xTMP = xIn
        self.al = []
        for theta in self.theta:
            xTMP = np.vstack( ( np.ones( np.shape(xIn)[1]), xTMP ) )
            self.al.append( xTMP )
            xTMP = theta.dot( xTMP )
            xTMP = sigmoid( xTMP )
        self.al.append( xTMP )
        return None

    def back_propagate(self, xOut, yData):
        #make sure the data has the correct layout
        assert np.shape(xOut)[0] == np.shape(yData)[0]
        xTMP = xOut - yData
        self.dl = []
        self.dl.append( xTMP )
        for ii in range( len(self.theta) ):
            idx = len(self.theta) - ii - 1
            thetaTMP = self.theta[idx]
            xTMP  = thetaTMP.T.dot( xTMP )
            xTMP = xTMP * self.al[ idx ] * (1. - self.al[ idx ] )
            xTMP  = np.delete( xTMP, 0, 0 )
            self.dl.append( xTMP )
        self.dl = self.dl[ ::-1 ]
        assert np.shape( self.al[0] )[0] - 1 == np.shape( self.dl[0] )[0]
        return None

    def calculate_derivatives(self):
        self.deri = []
        for ii in range( len(self.theta) ):
            ai = self.al[ ii ]
            di = self.dl[ ii + 1 ]
            xDim = np.shape(di)[0]
            yDim = np.shape(ai)[0]
            sumTMP = np.zeros( (xDim,yDim) )
            for jj in range( np.shape(ai)[1] ):
                sumTMP += np.outer( di.T[jj], ai.T[jj] )
            assert np.shape( sumTMP )[0] == xDim
            assert np.shape( sumTMP )[1] == yDim
            thetaTMP = self.theta[ii]
            #Do not regularize the bias term 
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

nLayers = np.array( [2,2,3] )
NN = NeuralNet( nLayers, 0 )
xTest = np.array( [ [0., 2.], [0., 2.] ] )
yData = np.array( [[1.], [0.], [0.] ] )
costOld = 100000000000.
costNew = 100000000.
while costNew < costOld - 0.001 :
    costOld = costNew
    NN.forward_propagate(xTest)
    xOut = NN.get_al( len(nLayers) - 1 )
    NN.back_propagate( xOut, yData )
    NN.calculate_derivatives()
    NN.update_theta(0.01)
    

    thetaList = []
    for ii in range (len(nLayers) - 1 ):
        thetaList.append( NN.get_theta( ii ) )
    costNew = cost( xOut, yData, 1., thetaList )
print costOld
