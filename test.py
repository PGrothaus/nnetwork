import nose
import numpy as np
import costfunction

def test_sigmoid():
    xArr = np.array( [0., 1000. ] )
    assert ( costfunction.sigmoid(xArr) == np.array( [0.5, 1.] ) ).all()


