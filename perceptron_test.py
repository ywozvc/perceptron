import perceptron
import perceptron_errors
import pytest
import numpy as np

@pytest.fixture
def new_perceptron():
    """
    Description
    ---
    create a new with a random n_tuple length perceptron for the other tests

    Returns
    ---
    perceptron object of random n_tuple length
    """
      
    return perceptron.Perceptron(np.random.randint(low=1, high=10))

def test_perceptron_object(new_perceptron):
    """
    Descripton
    ---
    tests a new perceptron object and make sure it has the necessary 
    elements not does not test to see if those things are of the right 
    type
    """
    assert new_perceptron.n_tuple is not None
    assert new_perceptron.learning_rate is not None
    assert new_perceptron.bias is not None

def test_n_tuple_boundedness():
    """
    Description
    ---
    tests the raising of the VectorError which occurs when a 
    n_tuple length is inputed that is not 1 or greater
    """
    with pytest.raises(perceptron_errors.VectorError):
        perceptron.Perceptron(-np.random.randint(low=0, high=10))

def test_sigmoidal_activation():
    """
    Description
    ---
    sigmoidal function must be either 0 or 1
    """
    for x in range(-100,100):
        assert perceptron.Perceptron.sigmoidal_activation(x) is 0 or 1
       # assert perceptron.Perceptron.sigmoidal_activation(x) <= 1
        #assert perceptron.Perceptron.sigmoidal_activation(x) >= 0

        
def test_weights(new_perceptron):
    """
    Description
    ---
    makes sure that the weights vector exists and is the same length as
    the vector plus the bias
    """
    assert len(new_perceptron.weights) > 0
    assert len(new_perceptron.weights)== new_perceptron.n_tuple+1



#def test_adjust(new_perceptron):#to do 
    #x_n = np.random.normal(loc=10, size=new_perceptron.n_tuple)
   # assert 
    
   
