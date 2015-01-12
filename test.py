import nose
import numpy as np
import nnetwork as nn
import costfunction

def test_sigmoid():
    xArr = np.array( [0., 1000. ] )
    assert ( costfunction.sigmoid(xArr) == np.array( [0.5, 1.] ) ).all()

def test_backprop():
    nLayers = np.array( [2,2,2] )
    NN = nn.NeuralNet( nLayers, 0. )
    theta0 = np.array( [ [0., 0.5, -0.5], [1., 1., 1.] ] )
    theta1 = np.array( [ [1., -2., 0.], [1., 2., 1.] ] )
    a0 = np.array( [ [1.], [0.5], [0.5] ] )
    a1 = np.array( [ [1.], [0.5], [0.5] ] ) 
    NN.set_theta( 0, theta0 )
    NN.set_theta( 1, theta1 )
    NN.forward_propagate( np.array( [ [1.], [1.] ] ) )
    NN.set_al( 0, a0 )
    NN.set_al( 1, a1 )
    yData  = np.array( [ [0.], [1.] ] )
    xOut   = np.array( [ [1.], [2.] ] )
    NN.back_propagate( xOut, yData )
    print NN.get_al( 2 )
    print NN.get_dl( 1 )
    print NN.get_dl( 0 )
    assert ( NN.get_dl( 0 ) == np.array( [ [1./16], [1./16] ] ) ).all()

def test_derivatives():
    nLayers = np.array( [2,2,2] )
    NN = nn.NeuralNet( nLayers, 0. )
    theta0 = np.array( [ [0., 0.5, -0.5], [1., 1., 1.] ] )
    theta1 = np.array( [ [1., -2., 0.], [1., 2., 1.] ] )
    a0 = np.array( [ [1.], [0.5], [0.5] ] )
    a1 = np.array( [ [1.], [0.5], [0.5] ] ) 
    NN.set_theta( 0, theta0 )
    NN.set_theta( 1, theta1 )
    NN.forward_propagate( np.array( [ [1.], [1.] ] ) )
    NN.set_al( 0, a0 )
    NN.set_al( 1, a1 )
    yData  = np.array( [ [0.], [1.] ] )
    xOut   = np.array( [ [1.], [2.] ] )
    NN.back_propagate( xOut, yData )
    NN.calculate_derivatives()
    d1 = NN.get_derivatives( 0 )
    d2 = NN.get_derivatives( 1 )
    print d1
    print d2
    assert ( d1 == np.array( [ [0, 0, 0], [0.25, 0.125, 0.125] ] ) ).all()
    assert ( d2 == np.array( [ [1, 0.5, 0.5], [1, 0.5, 0.5] ] ) ).all()

def test_update_theta():
    nLayers = np.array( [2,2,2] )
    NN = nn.NeuralNet( nLayers, 0. )
    theta0 = np.array( [ [0., 0.5, -0.5], [1., 1., 1.] ] )
    theta1 = np.array( [ [1., -2., 0.], [1., 2., 1.] ] )
    a0 = np.array( [ [1.], [0.5], [0.5] ] )
    a1 = np.array( [ [1.], [0.5], [0.5] ] ) 
    NN.set_theta( 0, theta0 )
    NN.set_theta( 1, theta1 )
    NN.forward_propagate( np.array( [ [1.], [1.] ] ) )
    NN.set_al( 0, a0 )
    NN.set_al( 1, a1 )
    yData  = np.array( [ [0.], [1.] ] )
    xOut   = np.array( [ [1.], [2.] ] )
    NN.back_propagate( xOut, yData )    
    NN.calculate_derivatives()
    NN.update_theta( 0.1 )
    print NN.get_theta(1)
    assert( NN.get_theta( 0 ) == np.array( [ 
                [0, 0.5, -0.5], [0.975, 0.9875, 0.9875] ] ) ).all()
    assert( NN.get_theta( 1 ) == np.array( [ 
                [0.9, -2.05, -0.05], [0.9, 1.95, 0.95] ] ) ).all()
