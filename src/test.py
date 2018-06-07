#! -*- coding:utf-8 -*-
'''
 concatenate all text files of corpus into one single file
'''

#from model import DistSimEvalModelGraph, ConvQAModelGraph
#from keras.models import Model
from layers import AttentionMatrixLayer, L2NormLayer, MaxOnASeqLayer, SumScoreLayer
from model import TestModelGraph
from keras.models import Model
from keras.layers import Dot 
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
    
    '''
    # eval value of tensor variables
    val = np.random.random((3,4))
    val_ = np.random.random((3,))
    #print(val)
    #print(val_)
    var = variable(value=val)
    var_ = variable(value=val_)
    #print(var)
    #c = MultiDimSftmxConstraint(axis=0)
    print(eval(var))
    '''
    
    q_seg = ['whats', 'this']
    q_len_denom = 1.0 / float(len(q_seg))
    q_len_denom_lst = [[q_len_denom,],]
    q_len_in = np.array(q_len_denom_lst, dtype=np.float32)
        
    model_graph = TestModelGraph(4)
    graph_in_units = model_graph.get_model_inputs()
    graph_out_units = model_graph.get_model_outputs()
    my_model = Model(inputs=graph_in_units, outputs=graph_out_units)
    
    my_model.compile(optimizer='adadelta', loss='binary_crossentropy')
    my_model.summary()
    #model.fit(x=np.array([[1.0,1.0,1.0,1.0,1.0]]), y=)
    for layer in my_model.layers:
        print(layer.get_weights())
    q = np.array([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
    a = np.array([[[-1.0, 0.0, 1.0, 1.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]]])
    x = [q, q_len_in, a]
    res = my_model.predict(x)
    print(res)
    