#! -*- coding:utf-8 -*-
'''

Model graph module

@author: LouisZBravo

'''
import config as cfg
from layers import MaxOnASeqLayer, SumScoreLayer, AttentionMatrixLayer, L2NormLayer, SumByAxisLayer
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dot, Concatenate, Dense, Dropout, Multiply, TimeDistributed, Softmax
from keras.layers import Permute, Reshape
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
        perspectives = cfg.ModelConfig.PERSPECTIVES
        
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
        '''
        gate_layer = TimeDistributed(Dense(units = wdim,
                                            activation = 'sigmoid',
                                            use_bias = True,
                                            ))
        _gate_for_q = gate_layer(_q_input)
        _gate_for_a = gate_layer(_a_input)
        _gated_q = Multiply()([_q_input, _gate_for_q])
        _gated_a = Multiply()([_a_input, _gate_for_a])
        _q_vec_normed = L2NormLayer()(_gated_q) # (sent len, wdim)
        _a_vec_normed = L2NormLayer()(_gated_a)
        _q_a_sim_mtx = Dot(axes=-1)([_q_vec_normed, _a_vec_normed])
        _q_words_best_match = MaxOnASeqLayer()(_q_a_sim_mtx)
        _q_match_score_sum = SumScoreLayer()(_q_words_best_match) # (1, )
        _q_match_score_ave = Multiply()([_q_match_score_sum, _q_len_denom_input]) # (1, )
        '''
        # multi-perspective version of the above mechanism
        _gate_layer_lst = []
        _q_gate_lst = []
        _a_gate_lst = []
        _gated_q_lst = []
        _gated_a_lst = []
        _vec_normed_q_lst = []
        _vec_normed_a_lst = []
        _sim_mtx_lst = []
        _word_atten_mtx_lst= []
        _q_words_best_match_lst = []
        _q_score_lst = []
        for i in range(perspectives):
            _gate_layer_lst.append(TimeDistributed(Dense(units = wdim,
                                                  activation = 'sigmoid',
                                                  use_bias = True,
                                                  )))
            _q_gate_lst.append(_gate_layer_lst[i](_q_input)) # (perspectives, sent len, wdim)
            _a_gate_lst.append(_gate_layer_lst[i](_a_input))
            _gated_q_lst.append(Multiply()([_q_input, _q_gate_lst[i]])) # (perspectives, sent len, wdim)
            _gated_a_lst.append(Multiply()([_a_input, _a_gate_lst[i]]))
            _vec_normed_q_lst.append(L2NormLayer()(_gated_q_lst[i])) # (perspectives, sent len, wdim)
            _vec_normed_a_lst.append(L2NormLayer()(_gated_a_lst[i]))
            _sim_mtx_lst.append(Dot(axes=-1)([_vec_normed_q_lst[i], _vec_normed_a_lst[i]])) # (perspectives, q_sent_len, a_sent_len)
            _word_atten_mtx_lst.append()
            _q_words_best_match_lst.append(MaxOnASeqLayer()(_sim_mtx_lst[i])) # (perspectives, q_sent_len)
            _sum_along_q = SumScoreLayer()(_q_words_best_match_lst[i]) # (1, )
            _q_score_lst.append(Multiply()([_sum_along_q, _q_len_denom_input])) # (perspectives, 1)
        
        feed_fwd_lst = _q_score_lst # a list of tensors to concatenate
        feed_fwd_tensr_len = perspectives # + 4
        # concatenate res dim: (?, )
        #_conc_res = Concatenate()([_q_pooled_maps_siam, _a_pooled_maps_siam, feat_match_1_to_2, feat_match_2_to_1, _add_feat_input])
        #_conc_res = Concatenate()([feat_match_1_to_2, feat_match_2_to_1, _add_feat_input])
        if len(feed_fwd_lst) > 1:
            _conc_res = Concatenate()(feed_fwd_lst)
        else:
            _conc_res = feed_fwd_lst[0]
        
        # hidden layer out dim: (?, )
        #_hid_gate_weight = Dense(units = feed_fwd_tensr_len, # always in accordance with input
        #                         activation = 'tanh',
        #                         use_bias = False,
        #                         kernel_regularizer = regularizers.l2(0.0001),
        #                         )(_conc_res)
        # dropout some units before computing softmax result
        #_dropped_hid_gate = Dropout(rate=0.5)(_hid_gate_weight)
        #_hid_res = Multiply()([_hid_gate_weight, _conc_res])
        
        #_res = Dense(units = 1,
        #             activation = 'sigmoid',
        #             use_bias = False,
        #             kernel_regularizer = regularizers.l2(0.0001),
        #             )(_dropped_hid_res)
        _res_gate_weight = Dense(units = feed_fwd_tensr_len, # always in accordance with input
                     activation = 'softmax',
                     use_bias = False,
                     kernel_regularizer = regularizers.l2(0.0001),
                     )(_conc_res)
        _res = Dot(axes=-1)([_res_gate_weight, _conc_res])
        self.graph_output_unit = _res
        
    def get_model_inputs(self):
        return self.graph_input_units
    
    def get_model_outputs(self):
        return self.graph_output_unit

class TestModelGraph(object):
    def __init__(self, wdim):
        # input dim: (sentence length, word dim)
        _q_input = Input(shape=(2, wdim))
        #_q_sent_len = _q_input.get_config['batch_input_shape'][1]
        _q_len_input = Input(shape=(1, )) # (1,), every elem: 1 / len(q) || 1 / len(set(q))
        _a_input = Input(shape=(3, wdim))
        #_a_sent_len = _a_input.get_config['batch_input_shape'][1]
        self.input_graph_unit = (_q_input, _q_len_input, _a_input)
        
        _q_vec_normed = L2NormLayer()(_q_input) #(q_len, wdim)
        _a_vec_normed = L2NormLayer()(_a_input)
        # output dim: (q sentence length, a sentence length)
        cos_sim_score_q_a = Dot(axes = -1)([_q_vec_normed, _a_vec_normed]) # (q_len, a_len)
        #atten_tsr = TimeDistributed(Multiply())([_q_vec_normed, _a_vec_normed])
        #sim_mtx_T = Permute((2, 1))(cos_sim_score_q_a)
        atten_mtx_sftmx = Softmax(axis = -2)(cos_sim_score_q_a) # (q_len, a_len)
        atten_ctxt_mtx = Dot(axes = -2)([atten_mtx_sftmx, _q_input])
        
        
        
        '''
        _q_words_best_match = MaxOnASeqLayer()(cos_sim_score_q_a)
        _q_match_score_sum = SumScoreLayer()(_q_words_best_match) # (1, )
        _q_match_score_ave = Multiply()([_q_match_score_sum, _q_len_input]) # (1, )
        
        hid_layer = Dense(units = 1,# + 4,
                         activation = 'linear',
                         kernel_initializer='ones',
                         use_bias = False, #kernel_regularizer = regularizers.l2(0.0001),
                         )
        hid_res = hid_layer(_q_match_score_ave)
        '''
        self.output_graph_unit = atten_ctxt_mtx#hid_res, _q_match_score_ave, _q_match_score_sum, _q_words_best_match, cos_sim_score_q_a
    
    def get_model_inputs(self):
        return self.input_graph_unit
    
    def get_model_outputs(self):
        return self.output_graph_unit
        