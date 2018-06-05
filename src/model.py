#! -*- coding:utf-8 -*-
'''

Model graph module

@author: LouisZBravo

'''
import config as cfg
from layers import MaxOnASeqLayer, SumScoreLayer
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dot, Concatenate, Dense, Dropout, Multiply
from keras import regularizers


class ConvQAModelGraph(object):
    def __init__(self, wdim):
        # hyperparameters
        conv_filter_len = cfg.ModelConfig.CONV_FILTER_LEN
        feat_map_num = cfg.ModelConfig.FEATURE_MAP_NUM
        
        # input dim: (sentence length, word dim)
        _q_input = Input(shape=(None, wdim))
        _a_input = Input(shape=(None, wdim))
        _add_feat_input = Input(shape=(4,))
        self.graph_input_units = (_q_input, _a_input, _add_feat_input)
        
        # feature map dim: (sent_len-filter_len+1, feat_map_num)
        siamese_conv_layer = Conv1D(input_shape = (None, wdim),
                                 filters = feat_map_num,
                                 kernel_size = conv_filter_len,
                                 activation = 'relu',
                                 kernel_regularizer = regularizers.l2(0.00001),
                                 )
        _q_feature_maps = siamese_conv_layer(_q_input)
        _a_feature_maps = siamese_conv_layer(_a_input)
        #_q_feature_maps = Conv1D(input_shape = (None, wdim),
        #                         filters = feat_map_num,
        #                         kernel_size = conv_filter_len,
        #                         activation = 'relu',
        #                         kernel_regularizer = regularizers.l2(0.00001),
        #                         )(_q_input)
        #_a_feature_maps = Conv1D(input_shape = (None, wdim),
        #                         filters = feat_map_num,
        #                         kernel_size = conv_filter_len,
        #                         activation = 'relu',
        #                         kernel_regularizer = regularizers.l2(0.00001),
        #                         )(_a_input)
                                 
        # pooling res dim: (feat_map_num, )
        _q_pooled_maps = GlobalMaxPooling1D()(_q_feature_maps)
        _a_pooled_maps = GlobalMaxPooling1D()(_a_feature_maps)
        
        # sentence match res dim: (1, )
        #sent_match_layer_0 = DotMatrixLayer(output_dim = feat_map_num)
        sent_match_layer_0 = Dense(units = feat_map_num,
                                   activation = None,
                                   use_bias = False,
                                   kernel_regularizer = regularizers.l2(0.0001),
                                   )
        sent_match_layer_1 = Dot(axes=-1)
        _qM_dot_res = sent_match_layer_0(_q_pooled_maps)
        _sent_match_res = sent_match_layer_1([_qM_dot_res, _a_pooled_maps])
        
        # concatenate res dim: (2*feat_map_num+5, )
        _conc_res = Concatenate()([_q_pooled_maps, _sent_match_res, _a_pooled_maps, _add_feat_input])
        
        # hidden layer out dim: (2*feat_map_num+5, )
        _hid_res = Dense(units = 2 * feat_map_num + 5,
                         activation = 'tanh',
                         use_bias = True,
                         kernel_regularizer = regularizers.l2(0.0001),
                         )(_conc_res)
                         
        # dropout some units before computing softmax result
        _dropped_hid_res = Dropout(rate=0.5)(_hid_res)
                        
        # softmax binary classifier out dim: (2, )
        """_bin_res = Dense(units = 2,
                         activation = 'softmax',
                         use_bias = False,
                         kernel_regularizer = regularizers.l2(0.0001),
                         )(_dropped_hid_res)
        
        self.graph_output_unit = _bin_res[:, 0]"""
        
        _res = Dense(units = 1,
                     activation = 'sigmoid',
                     use_bias = False,
                     kernel_regularizer = regularizers.l2(0.0001),
                     )(_dropped_hid_res)
        self.graph_output_unit = _res
        
    def get_model_inputs(self):
        return self.graph_input_units
    
    def get_model_outputs(self):
        return self.graph_output_unit

class DistSimEvalModelGraph(object):
    def __init__(self, wdim):
        # input dim: (sentence length, word dim)
        _q_input = Input(shape=(None, wdim))
        _q_len_input = Input(shape=(None,)) # (q sentence len,), every elem: 1 / len(q) || 1 / len(set(q))
        _a_input = Input(shape=(None, wdim))
        self.input_graph_unit = (_q_input, _q_len_input, _a_input)
        
        vec_cos_sim_calc_layer = Dot(axes=-1)
        # output dim: (q sentence length, a sentence length)
        cos_sim_score_q_a = vec_cos_sim_calc_layer([_q_input, _a_input])
        
        # output dim: (q sentence length,))
        max_cos_sim_score_4_q = MaxOnASeqLayer()(cos_sim_score_q_a)
        score_divided = Multiply()([max_cos_sim_score_4_q, _q_len_input])
        
        # output dim: (1,) each batch one
        score_ave = SumScoreLayer()(score_divided)
        
        self.output_graph_unit = score_ave
    
    def get_model_inputs(self):
        return self.input_graph_unit
    
    def get_model_outputs(self):
        return self.output_graph_unit
        