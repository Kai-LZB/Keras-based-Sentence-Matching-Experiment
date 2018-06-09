#! -*- coding:utf-8 -*-
'''

customized layer design

@author: LouisZBravo

'''

from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
from keras.backend import max as tensor_max
from keras.backend import sum as tensor_sum
from keras.backend import exp, l2_normalize

from keras.constraints import Constraint

class AttentionMatrixLayer(Layer):
    """
     perform feature attention mechanism from input feature vector to output feature
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AttentionMatrixLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                            shape=(input_shape[1], self.output_dim), # q_len, a_len
                            initializer = 'ones', # 'random_uniform'
                            regularizer = regularizers.l2(0.0001),
                            trainable=True,
                            ) # input_shape[1] is weight vector(each col, for input vec) length
        super(AttentionMatrixLayer, self).build(input_shape)
    def call(self, x):
        # for-attention multi-dimensional softmax
        #max_for_each_axis = tensor_max(self.kernel, axis=0, keepdims=True) # 1, a len
        #target_to_be_exp = self.kernel - max_for_each_axis # for numerical stability, thanks to raingo @ gist.github.com/raingo/a5808fe356b8da031837
        #exp_tensor = exp(target_to_be_exp)
        #norm = tensor_sum(exp_tensor, axis=0, keepdims=True)
        #target_w = exp_tensor / norm
        
        # for-attention l2-normalization
        target_w = l2_normalize(self.kernel, axis = 0)
        
        # for-attention raw transformation
        #target_w = self.kernel
        return K.dot(x, target_w)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim) # I would say in_shape[0] is batch dim
    

        
class L2NormLayer(Layer):
    """
     perform l2 normalization for vector
    """
    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(L2NormLayer, self).build(input_shape)
    
    def call(self, x):
        return l2_normalize(x, axis = -1)
    
    def compute_output_shape(self, input_shape):
        return input_shape # I would say in_shape[0] is batch dim


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
'''
class KMaxScoreLayer(Layer):
    """
     Given tensor of size (s_len,)
     this layer finds the best k score for each match res in s, calc the ave of them 
    """
    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(KMaxScoreLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(KMaxScoreLayer, self).build(input_shape)
        
    def call(self, x):
        return tensor_sum(x, axis=-1, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)  # (batch_idx, 1)'''
    
class SumScoreLayer(Layer):
    """
     Given tensor of size (s_len,)
     this layer sums score for each word in s
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
    
class SumByAxisLayer(Layer):
    """
     Given tensor of any size
     this layer sums along the specified axis
    """
    def __init__(self, axis = -1, keepdims=True, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super(SumByAxisLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(SumByAxisLayer, self).build(input_shape)
        
    def call(self, x):
        return tensor_sum(x, axis=self.axis, keepdims=self.keepdims)
    
    def compute_output_shape(self, input_shape):
        axis = self.axis
        if self.keepdims and axis == -1:
            return input_shape[:axis] + (1,)
        elif self.keepdims:
            return input_shape[:axis] + (1,) + input_shape[axis+1:]
        elif axis == -1:
            return input_shape[:axis]
        else:
            return input_shape[:axis] + input_shape[axis+1:]
    
class MultiDimSftmxConstraint(Constraint):
    '''
     A multi-dimensional constraint on optimization
    '''
    def __init__(self, axis=0):
        self.axis = axis
        
    def __call__(self, w):
        max_for_each_axis = tensor_max(w, axis=self.axis, keepdims=True)
        target_to_be_exp = w - max_for_each_axis # for numerical stability, thanks to raingo @ gist.github.com/raingo/a5808fe356b8da031837
        exp_tensor = exp(target_to_be_exp)
        norm = tensor_sum(exp_tensor, axis=self.axis, keepdims=True)
        res_sftmx = exp_tensor / norm
        return res_sftmx
    
    def get_config(self):
        return {'axis': self.axis}
