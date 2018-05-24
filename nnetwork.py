import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


class NeuralNet:
    def __init__(self, nNodesInLayer, regP):
        assert type(nNodesInLayer) is np.ndarray
        assert len(nNodesInLayer) > 1
        self.nLayers = len(nNodesInLayer)
        self.nNodes = np.sum(nNodesInLayer)
        self.NodesInLayer = nNodesInLayer
        self.theta = []
        self.al = []
        self.dl = []
        self.deri = []
        self.regParam = regP
        for ii in range(self.nLayers - 1):
            nIn = self.NodesInLayer[ii]
            nOut = self.NodesInLayer[ii + 1]
            self.theta.append(np.random.normal(0., .5,
                                               size=(nOut, nIn + 1)))
        print ''
        print 'NEURAL NETWORK created!'
        print self.nLayers, 'layers in total (including In- and Output-layer)'
        print len(self.theta), 'parameter matrices'
        print ''

    def get_nodes_in_layer(self, n):
        return self.NodesInLayer[n]

    def get_theta(self, n):
        assert n < self.nLayers - 1,  'n too large. Not so many '\
            'parameter matrices'
        return self.theta[n]

    def get_al(self, n):
        return self.al[n]

    def set_al(self, n, ai):
        self.al[n] = ai

    def get_dl(self, n):
        return self.dl[n]

    def get_derivatives(self, n):
        assert n < self.nLayers - 1,  'n too large. Not so many layers'
        return self.deri[n]

    def forward_propagate(self, xIn):
        # Need to add bias term to xIN -> vstack
        assert np.shape(xIn)[0] + 1 == np.shape(self.theta[0])[1]
        xTMP = xIn
        self.al = []
        for theta in self.theta:
            xTMP = np.vstack((np.ones(np.shape(xIn)[1]), xTMP))
            self.al.append(xTMP)
            xTMP = np.dot(theta, xTMP)
            xTMP = sigmoid(xTMP)
        self.al.append(xTMP)
        return None

    def back_propagate(self, xOut, yData):
        # make sure the data has the correct layout
        assert np.shape(xOut)[0] == np.shape(yData)[0]
        xTMP = np.array(xOut - yData)
        self.dl = []
        self.dl.append(xTMP)
        # could supress last loop, because d0 is not used fo calc derivatives
        for ii in range(len(self.theta)):
            idx = len(self.theta) - ii - 1
            xTMP = np.dot(self.theta[idx].T, xTMP)
            xTMP = xTMP * self.al[idx] * (1. - self.al[idx])
            xTMP = np.delete(xTMP, 0, 0)
            self.dl.append(xTMP)
        self.dl = self.dl[::-1]
        assert np.shape(self.al[0])[0] - 1 == np.shape(self.dl[0])[0]
        return None

    def calculate_derivatives(self):
        self.deri = []
        for ii in range(len(self.theta)):
            ai = self.al[ii]
            di = self.dl[ii + 1]
            m = np.shape(ai)[1]
            xDim = np.shape(self.theta[ii])[0]
            yDim = np.shape(self.theta[ii])[1]
            sumTMP = np.einsum('ik, jk', di, ai)
            assert np.shape(sumTMP) == (xDim, yDim)
            thetaTMP = np.array(self.theta[ii])
            # Do not regularize the bias term
            thetaTMP.T[0] = 0
            self.deri.append(1. * sumTMP / m +
                             self.regParam * thetaTMP / (2. * m))
        return None

    def update_theta(self, epsilon):
        for ii in range(len(self.theta)):
            self.theta[ii] = (self.theta[ii] - epsilon * self.deri[ii])
        return None

    def train(self, xIn, yData, epsilon=1.2, precision=0.001):
        self.forward_propagate(xIn)
        xOut = self.get_al(self.nLayers - 1)
        self.back_propagate(xOut, yData)
        self.calculate_derivatives()
        costOld = self.cost(xOut, yData)
        costNew = np.array(costOld)
        TRAIN = True
        count = 0
        while TRAIN:
            costOld = np.array(costNew)
            if (0 == count % 10):
                print count, epsilon, costOld
            self.update_theta(epsilon)
            self.forward_propagate(xIn)
            xOut = self.get_al(self.nLayers - 1)
            costNew = self.cost(xOut, yData)
            if (costNew < costOld).all():
                self.back_propagate(xOut, yData)
                self.calculate_derivatives()
                if ((costNew + precision) > costOld).any():
                    TRAIN = False
            elif (costNew > costOld).any():
                self.update_theta(-1. * epsilon)
                costNew = costOld
                epsilon = epsilon / 2.
                if epsilon < 0.04:
                    TRAIN = False
            count += 1
        print 'Final Cost:', costNew
        return costNew

    def cost(self, xOut, yData):
        xOut = np.where(xOut == 0., 10 ** (-10),      xOut)
        xOut = np.where(xOut == 1., 1. - 10 ** (-10), xOut)
        m = np.shape(xOut)[1]
        sumTheta = 0.
        for theta in self.theta:
            theta = np.delete(theta, 0, 1)
            sumTheta += np.sum(theta * theta)
        sumCost = np.array([- yData * np.log(xOut)])
        sumCost = sumCost - (1. - yData) * np.log(1. - xOut)
        sumCost = 1. / m * np.sum(sumCost)
        sumTheta = 1. / 2 / m * sumTheta
        sumTheta = self.regParam * sumTheta
        return sumCost + sumTheta

    def set_theta(self, n, thetaIn):
        assert isinstance(thetaIn, np.ndarray)
        assert thetaIn.shape == self.theta[n].shape
        self.theta[n] = thetaIn

    def check_grad(self, n, ii, jj, xIn, yData, diff=0.0001):
        thetaSRC = np.array(self.theta[n])
        thetaNew = np.array(self.theta[n])
        thetaNew[(ii, jj)] = thetaNew[(ii, jj)] + diff
        grad = self.get_derivatives(n)[ii][jj]
        self.forward_propagate(xIn)
        xOut = self.get_al(self.nLayers - 1)
        costOld = self.cost(xOut, yData)
        self.set_theta(n, thetaNew)
        self.forward_propagate(xIn)
        xOut = self.get_al(self.nLayers - 1)
        costNew = self.cost(xOut, yData)
        gradTest = (costNew - costOld) / (diff)
        self.set_theta(n, thetaSRC)
        print ' Results of gradient checking: '
        print grad, gradTest
        print ''

    def feature_scaling(self, xIn):
        m = np.shape(xIn)[1]
        featureMax = np.max(xIn, axis=1)
        featureMax = np.reshape(featureMax, (np.shape(xIn)[0], 1))
        featureMax = np.repeat(featureMax, m, axis=1)
        featureMin = np.min(xIn, axis=1)
        featureMin = np.reshape(featureMin, (np.shape(xIn)[0], 1))
        featureMin = np.repeat(featureMin, m, axis=1)
        featureMax = np.where(featureMin == featureMax, 1., featureMax)
        xIn = (xIn - featureMin) / (featureMax - featureMin)
        return xIn

    def predict(self, xIn):
        self.forward_propagate(xIn)
        yPred = self.get_al(self.nLayers - 1)
        maxArr = np.max(yPred, axis=0)
        maxArr = np.reshape(maxArr, (1, len(maxArr)))
        yPred = yPred - np.repeat(maxArr, np.shape(yPred)[0], axis=0)
        yPred = np.where(yPred < 0., 0., 1.)
        yPred = yPred.T
        yPred = np.nonzero(yPred)[1]
        return yPred

    def write_to_file(self, filename):
        with open(filename, "w") as f:
            pass
        with open(filename, "a") as f:
            for mat in self.theta:
                for ii in range(np.shape(mat)[0]):
                    for jj in range(np.shape(mat)[1]):
                        f.write(str(mat[ii][jj]) + '\n')
