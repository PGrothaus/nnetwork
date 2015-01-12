import numpy as np
import nnetwork as nn

DataSize = 20000
TestSize = 20000
outDim = 10

#init neural network
nLayers = np.array( [784,   outDim ] )
NN = nn.NeuralNet( nLayers, .5 )

data = np.loadtxt( "data/train_1.txt", delimiter=',' )
label = np.array( data[ : , 0 ] ,dtype = int)
pixelData = np.array( data[ : , 1 : ] )

data = np.loadtxt( "data/train_2.txt", delimiter=',' )
labelTest = np.array( data[ : , 0 ] ,dtype = int)
pixelTest = np.array( data[ : , 1 : ] )

#Bring into correct shape
pixelData = pixelData.T
pixelTest = pixelTest.T

DataID = np.arange( DataSize )
yData  = np.zeros( ( outDim, DataSize ) )
TestID = np.arange( TestSize )
yTest  = np.zeros( ( outDim, TestSize ) )

yData[ label, DataID ] = 1
yTest[ labelTest, TestID ] = 1

#scale Data
print 'rescale...'
pixelData = NN.feature_scaling( pixelData )
pixelTest = NN.feature_scaling( pixelTest )

#Train NN
print ''
print 'Start training NN'
NN.train( pixelData, yData, precision = 0.1 )

#Evaluate Training Accuracy
yPred = NN.predict( pixelData )
diff  = np.where( yPred == label, 1, 0  )
Ncorr = np.sum( diff )
print 'Training accuracy:', 1. * Ncorr / len( label )

#Predict next labels:
yPred = NN.predict( pixelTest )
diff  = np.where( yPred == labelTest, 1, 0  )
Ncorr = np.sum( diff )
print 'Prediction accuracy:', 1. * Ncorr / len( labelTest )

#save output
NN.write_to_file( "NN_500_05.txt" )
print ''

data = np.loadtxt( "data/test.csv", delimiter=',' )
Testdata = np.array( data[ : , : ] ,dtype = int)
Testdata = Testdata.T
yPred = NN.predict( Testdata )
f = open( "prediction.txt", "w" )
f.close()
f = open( "prediction.txt", "a" )
for label in yPred:
    f.write( str(label)+'\n' )
f.close()
