#! -*- coding:utf-8 -*-
'''

Model graph module

@author: LouisZBravo

'''
import config as cfg
from layers import MaxOnASeqLayer, SumScoreLayer, AttentionMatrixLayer, L2NormLayer
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dot, Concatenate, Dense, Dropout, Multiply, TimeDistributed, Activation
from keras import regularizers


class ConvQAModelGraph(object):
    def __init__(self, wdim):
        # hyperparameters
        conv_filter_len_siam = cfg.ModelConfig.CONV_FILTER_LEN_SIAM
        feat_map_num_siam = cfg.ModelConfig.FEATURE_MAP_NUM_SIAM
        conv_filter_len_1 = cfg.ModelConfig.CONV_FILTER_LEN_1
        feat_map_num_1 = cfg.ModelConfig.FEATURE_MAP_NUM_1
        conv_filter_len_2 = cfg.ModelConfig.CONV_FILTER_LEN_2
        feat_map_num_2 = cfg.ModelConfig.FEATURE_MAP_NUM_2
        
        # input dim: (sentence length, word dim)
        _q_input = Input(shape=(None, wdim))
        _a_input = Input(shape=(None, wdim))
        _q_len_denom_input = Input(shape=(1,))
        _a_len_denom_input = Input(shape=(1,))
        _add_feat_input = Input(shape=(4,))
        self.graph_input_units = (_q_input, _a_input, _q_len_denom_input, _a_len_denom_input, _add_feat_input)
        
        # siamese convolution layer out dim: (sentence length, feat map num)
        #siamese_conv_layer = Conv1D(input_shape = (None, wdim),
        #                         filters = feat_map_num_siam,
        #                         kernel_size = conv_filter_len_siam,
        #                         padding='same',
        #                         activation = 'relu',
        #                         kernel_regularizer = regularizers.l2(0.00001),
        #                         )
        #_q_feature_maps_siam = siamese_conv_layer(_q_input)
        #_a_feature_maps_siam = siamese_conv_layer(_a_input)
        
        # independent convolution layer 1 out dim: (sentence length_1, feat map num_1)
        # independent convolution layer 2 out dim: (sentence length_2, feat map num_2)
        #_q_feature_maps_indep = Conv1D(input_shape = (None, wdim),
        #                         filters = feat_map_num_1,
        #                         kernel_size = conv_filter_len_1,
        #                         padding='same',
        #                         activation = 'relu',
        #                         kernel_regularizer = regularizers.l2(0.00001),
        #                         )(_q_input)
        #_a_feature_maps_indep = Conv1D(input_shape = (None, wdim),
        #                         filters = feat_map_num_2,
        #                         kernel_size = conv_filter_len_2,
        #                         padding='same',
        #                         activation = 'relu',
        #                         kernel_regularizer = regularizers.l2(0.00001),
        #                         )(_a_input)
                                 
        # siamese pooling res dim: (feat_map_num_1, )
        #_q_pooled_maps_siam = GlobalMaxPooling1D()(_q_feature_maps_siam)
        #_a_pooled_maps_siam = GlobalMaxPooling1D()(_a_feature_maps_siam)
        
        # q pooling indep res dim: (feat_map_num_1, )
        # a pooling indep res dim: (feat_map_num_2, )
        #_q_pooled_maps_indep = GlobalMaxPooling1D()(_q_feature_maps_indep)
        #_a_pooled_maps_indep = GlobalMaxPooling1D()(_a_feature_maps_indep)
        
        # bilateral feature attention
        # feat_1 -> feat_2 attention res dim: (feat_map_2, )
        # feat_2 -> feat_1 attention res dim: (feat_map_1, )
        #attention_layer_1_to_2 = AttentionMatrixLayer(output_dim = feat_map_num_2)
        #attention_layer_2_to_1 = AttentionMatrixLayer(output_dim = feat_map_num_1)
        #attentive_q_to_a_feat = attention_layer_1_to_2(_q_pooled_maps_indep)
        #attentive_a_to_q_feat = attention_layer_2_to_1(_a_pooled_maps_indep)
        
        # norm before dot
        #normed_q_to_a_feat = L2NormLayer()(attentive_q_to_a_feat)
        #normed_a_to_q_feat = L2NormLayer()(attentive_a_to_q_feat)
        #normed_q_feat = L2NormLayer()(_q_pooled_maps_indep)
        #normed_a_feat = L2NormLayer()(_a_pooled_maps_indep)
        # dot matching, res dim: (1, )
        #feat_match_layer = Dot(axes=-1)
        #feat_match_1_to_2 = feat_match_layer([normed_q_to_a_feat, normed_a_feat])
        #feat_match_2_to_1 = feat_match_layer([normed_a_to_q_feat, normed_q_feat])
        
        # distributional similarity, out dim: (1, )
        #_atten_q = TimeDistributed(AttentionMatrixLayer(output_dim=wdim))(_q_input)
        #gate_layer = TimeDistributed(Dense(units = wdim,
        #                                    activation = 'sigmoid',
        #                                    use_bias = True,
        #                                    ))
        #_gate_for_q = gate_layer(_q_input)
        #_gate_for_a = gate_layer(_a_input)
        #_gated_q = Multiply()([_q_input, _gate_for_q])
        #_gated_a = Multiply()([_a_input, _gate_for_a])
        #_gated_q = Multiply()([_atten_q, _gate_for_q])
        _q_vec_normed = L2NormLayer()(_q_input) # (sent len, wdim)
        _a_vec_normed = L2NormLayer()(_a_input)
        vec_cos_sim_calc_layer = Dot(axes=-1)
        _q_a_sim_mtx = vec_cos_sim_calc_layer([_q_vec_normed, _a_vec_normed]) # (q_sent_len, a_sent_len)
        _q_words_best_match = MaxOnASeqLayer()(_q_a_sim_mtx) # (q_sent_len, )
        _q_match_score_sum = SumScoreLayer()(_q_words_best_match) # (1, )
        _q_match_score_ave = Multiply()([_q_match_score_sum, _q_len_denom_input]) # (1, )
        
        
        # concatenate res dim: (6, )
        #_conc_res = Concatenate()([_q_pooled_maps_siam, _a_pooled_maps_siam, feat_match_1_to_2, feat_match_2_to_1, _add_feat_input])
        #_conc_res = Concatenate()([feat_match_1_to_2, feat_match_2_to_1, _add_feat_input])
        _conc_res = Concatenate()([_q_match_score_ave, ])
        
        # hidden layer out dim: (6, )
        _hid_res = Dense(units = 1,#2 + 4,
                         activation = 'tanh',
                         use_bias = True,
                         kernel_regularizer = regularizers.l2(0.0001),
                         )(_conc_res)
                         
        # dropout some units before computing softmax result
        #_dropped_hid_res = Dropout(rate=0.5)(_hid_res)
                        
        # softmax binary classifier out dim: (2, )
        """_bin_res = Dense(units = 2,
                         activation = 'softmax',
                         use_bias = False,
                         kernel_regularizer = regularizers.l2(0.0001),
                         )(_dropped_hid_res)
        
        self.graph_output_unit = _bin_res[:, 0]"""
        
        #_res = Dense(units = 1,
        #             activation = 'sigmoid',
        #             use_bias = False,
        #             kernel_regularizer = regularizers.l2(0.0001),
        #             )(_dropped_hid_res)
        _res = Dense(units = 1,
                     activation = 'sigmoid',
                     use_bias = False,
                     kernel_regularizer = regularizers.l2(0.0001),
                     )(_hid_res)
        self.graph_output_unit = _res
        
    def get_model_inputs(self):
        return self.graph_input_units
    
    def get_model_outputs(self):
        return self.graph_output_unit

class TestModelGraph(object):
    def __init__(self, wdim):
        # input dim: (sentence length, word dim)
        _q_input = Input(shape=(None, wdim))
        _q_len_input = Input(shape=(1, )) # (1,), every elem: 1 / len(q) || 1 / len(set(q))
        _a_input = Input(shape=(None, wdim))
        self.input_graph_unit = (_q_input, _q_len_input, _a_input)
        
        _q_vec_normed = L2NormLayer()(_q_input)
        _a_vec_normed = L2NormLayer()(_a_input)
        vec_cos_sim_calc_layer = Dot(axes=-1)
        # output dim: (q sentence length, a sentence length)
        cos_sim_score_q_a = vec_cos_sim_calc_layer([_q_vec_normed, _a_vec_normed])
        _q_words_best_match = MaxOnASeqLayer()(cos_sim_score_q_a)
        _q_match_score_sum = SumScoreLayer()(_q_words_best_match) # (1, )
        _q_match_score_ave = Multiply()([_q_match_score_sum, _q_len_input]) # (1, )
        
        self.output_graph_unit = _q_match_score_ave, _q_match_score_sum, _q_words_best_match, cos_sim_score_q_a
    
    def get_model_inputs(self):
        return self.input_graph_unit
    
    def get_model_outputs(self):
        return self.output_graph_unit
        