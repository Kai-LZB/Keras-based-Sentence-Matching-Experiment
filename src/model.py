#! -*- coding:utf-8 -*-
'''

Model graph module

@author: LouisZBravo

'''
import config as cfg
from layers import MaxOnASeqLayer, SumScoreLayer, AttentionMatrixLayer
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dot, Concatenate, Dense, Dropout, Multiply
from keras import regularizers


class ConvQAModelGraph(object):
    def __init__(self, wdim):
        # hyperparameters
        conv_filter_len_1 = cfg.ModelConfig.CONV_FILTER_LEN_1
        feat_map_num_1 = cfg.ModelConfig.FEATURE_MAP_NUM_1
        conv_filter_len_2 = cfg.ModelConfig.CONV_FILTER_LEN_2
        feat_map_num_2 = cfg.ModelConfig.FEATURE_MAP_NUM_2
        
        # input dim: (sentence length, word dim)
        _q_input = Input(shape=(None, wdim))
        _a_input = Input(shape=(None, wdim))
        _add_feat_input = Input(shape=(4,))
        self.graph_input_units = (_q_input, _a_input, _add_feat_input)
        
        _q_feature_maps = Conv1D(input_shape = (None, wdim),
                                 filters = feat_map_num_1,
                                 kernel_size = conv_filter_len_1,
                                 padding='same',
                                 activation = 'relu',
                                 kernel_regularizer = regularizers.l2(0.00001),
                                 )(_q_input)
        _a_feature_maps = Conv1D(input_shape = (None, wdim),
                                 filters = feat_map_num_2,
                                 kernel_size = conv_filter_len_2,
                                 padding='same',
                                 activation = 'relu',
                                 kernel_regularizer = regularizers.l2(0.00001),
                                 )(_a_input)
                                 
        # pooling res dim: (feat_map_num, )
        _q_pooled_maps = GlobalMaxPooling1D()(_q_feature_maps)
        _a_pooled_maps = GlobalMaxPooling1D()(_a_feature_maps)
        
        # bilateral feature attention
        # feat_1 -> feat_2 attention res dim: (feat_map_2, )
        # feat_2 -> feat_1 attention res dim: (feat_map_1, )
        attention_layer_1_to_2 = AttentionMatrixLayer(output_dim = feat_map_num_2)
        attention_layer_2_to_1 = AttentionMatrixLayer(output_dim = feat_map_num_1)
        attentive_q_to_a_feat = attention_layer_1_to_2(_q_pooled_maps)
        attentive_a_to_q_feat = attention_layer_2_to_1(_a_pooled_maps)
        
        # vec dot vec, res dim: (1, )
        feat_match_layer = Dot(axes=-1)
        feat_match_1_to_2 = feat_match_layer([attentive_q_to_a_feat, _a_pooled_maps])
        feat_match_2_to_1 = feat_match_layer([attentive_a_to_q_feat, _q_pooled_maps])
        
        # concatenate res dim: (2*feat_map_num+5, )
        _conc_res = Concatenate()([_q_pooled_maps, _a_pooled_maps, feat_match_1_to_2, feat_match_2_to_1, _add_feat_input])
        
        # hidden layer out dim: (2*feat_map_num+5, )
        _hid_res = Dense(units = feat_map_num_1 + feat_map_num_2 + 2 + 4,
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
        