#! -*- coding:utf-8 -*-
'''
 concatenate all text files of corpus into one single file
'''

#from model import DistSimEvalModelGraph, ConvQAModelGraph
#from keras.models import Model
from layers import AttentionMatrixLayer, L2NormLayer
from keras.constraints import non_neg
from keras.backend import variable, eval, dot
import numpy as np

if __name__ == '__main__':
    '''model_graph = DistSimEvalModelGraph(50)
    #model_graph = ConvQAModelGraph(50)
    
    model_in_unit = model_graph.get_model_inputs()
    model_out_unit = model_graph.get_model_outputs()
    
    my_model = Model(inputs=model_in_unit, outputs=model_out_unit)
    loss_func = 'binary_crossentropy'
    optm = 'adadelta'
    my_model.compile(optimizer=optm, loss=loss_func, metrics=['accuracy'])'''
    
    val = np.random.random((3,4))
    val_ = np.random.random((3,))
    #print(val)
    #print(val_)
    var = variable(value=val)
    var_ = variable(value=val_)
    #print(var)
    #c = MultiDimSftmxConstraint(axis=0)
    print(eval(var))
    
    from keras.models import Sequential
    model = Sequential()
    model.add(AttentionMatrixLayer(input_shape=(5,), output_dim=10))
    model.add(L2NormLayer())
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    #model.fit(x=np.array([[1.0,1.0,1.0,1.0,1.0]]), y=)
    for layer in model.layers:
        print(layer.get_weights())
    res = model.predict(np.array([[1.0,1.0,1.0,1.0,1.0]]))
    print(res)
    