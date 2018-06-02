#! -*- coding:utf-8 -*-
'''
 concatenate all text files of corpus into one single file
'''

from model import DistSimEvalModelGraph, ConvQAModelGraph
from keras.models import Model

if __name__ == '__main__':
    model_graph = DistSimEvalModelGraph(50)
    #model_graph = ConvQAModelGraph(50)
    
    model_in_unit = model_graph.get_model_inputs()
    model_out_unit = model_graph.get_model_outputs()
    
    my_model = Model(inputs=model_in_unit, outputs=model_out_unit)
    loss_func = 'binary_crossentropy'
    optm = 'adadelta'
    my_model.compile(optimizer=optm, loss=loss_func, metrics=['accuracy'])
    
    print(my_model.summary())