#! -*- coding:utf-8 -*-
'''

customized layer design

@author: LouisZBravo

'''

from keras import backend as K
from keras.engine.topology import Layer
from keras.backend import max as tensor_max
from keras.backend import max as tensor_sum

class DotMatrixLayer(Layer):
    """
     This is a layer that can be substituted by a Dense layer
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DotMatrixLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True,
                                      ) # in_shape[1] is vector length
        super(DotMatrixLayer, self).build(input_shape)
        
    def call(self, x):
        return K.dot(x, self.kernel)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim) # I would say in_shape[0] is batch dim
        
        
class MaxOnASeqLayer(Layer):
    """
     Given tensor (q_len, a_len)
     this layer calc for each word in q the best match in a, getting the max cos sim score
    """
    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(MaxOnASeqLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MaxOnASeqLayer, self).build(input_shape)
        
    def call(self, x):
        return tensor_max(x, axis=-1, keepdims=False)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])  # (batch_idx, q_len)

class SumScoreLayer(Layer):
    """
     Given tensor (q_len)
     this layer sums best match score for each word in q, calc the general score 
    """
    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(SumScoreLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(SumScoreLayer, self).build(input_shape)
        
    def call(self, x):
        return tensor_sum(x, axis=-1, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)  # (batch_idx, 1)
    