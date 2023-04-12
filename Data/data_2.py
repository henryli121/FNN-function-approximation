import numpy as np
import math

def myfunction(x1, x2):
    return (math.sin(5*x1)*math.cos(5*x2)+1)/2

x1 = np.random.uniform(0,1,50000)
x2 = np.random.uniform(0,1,50000)
y = [myfunction(x1[k],x2[k]) for k in range(len(x1))]

t1 = np.random.uniform(0,1,10000)
t2 = np.random.uniform(0,1,10000)
ty = [myfunction(t1[k],t2[k]) for k in range(len(t1))]

def points():
    x1 = t1
    y1 = t2
    z1 = ty
    return(x1, y1, z1)

def load_data():
    training_inputs = [np.reshape(list(zip(x1,x2)),(len(x1),2))]
    training_data = list(zip(training_inputs,[np.array(y)]))

    test_inputs = [np.reshape(list(zip(t1,t2)),(len(t1),2))]
    test_data = list(zip(test_inputs,[np.array(ty)]))
    return (training_data[0], test_data[0])

def load_data_wrapper():
    tr_d, te_d = load_data()
    training_inputs = [np.reshape(x, (2, 1)) for x in tr_d[0]]
    training_results = [y for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    test_inputs = [np.reshape(x, (2, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, test_data)
