#! -*- coding:utf-8 -*-
'''

Model configuration:
Hyper parameters: to business module directly
Static parameters for dynamic control: to control logic module

@author: LouisZBravo

'''
class DirConfig(object):
    DATA_DIR = '../data/'
    DATA_RAW_DIR = '../data/raw/'
    DATA_CACHED_DIR = '../data/cached/'
    # HITNLP_WV_PATH = DATA_DIR + 'HITNLP' + '_w' + 'vec.db'
    CORPUS_PATH_DICT = {'MIXED_CHN': DATA_RAW_DIR + 'mixed_corpus',
                   'ENG_TEST': DATA_RAW_DIR + 'text8',
                   'CHN_TEST': DATA_RAW_DIR + 'test_chn_corpus',
                   '_PRETRAINED': None,
                   }
    CLEAN_CORPUS_PATH_DICT = {'MIXED_CHN': DATA_CACHED_DIR + 'mixed_corpus_clean',
                         'ENG_TEST': DATA_CACHED_DIR + 'text8_clean',
                         'CHN_TEST': DATA_CACHED_DIR + 'test_chn_corpus_clean',
                         '_PRETRAINED': None,
                         }
    # for cc path in word2vec directory, see dst_corpus_path in model_control module, exec_ function
    CLEAN_CORPUS_FILENAME_DICT = {'MIXED_CHN': 'mixed_corpus_clean',
                             'ENG_TEST': 'text8_clean',
                             'CHN_TEST': 'test_chn_corpus_clean',
                             '_PRETRAINED': None,
                             }
    QA_DATA_FILENAME_DICT = {'HITNLP': {'DEV': 'develop.data', 'TRAIN': 'training.data', 'TEST': 'randomed_labeled_testing.data', 'NAH': ''},
                             }
    QA_DATA_PATH_DICT = {'HITNLP': {'DEV': DATA_RAW_DIR + 'develop.data', 'TRAIN': DATA_RAW_DIR + 'training.data', 'TEST': DATA_RAW_DIR + 'randomed_labeled_testing.data', 'NAH': ''},
                         }
    #CLEAN_QA_DATA_PATH_DICT = {'HITNLP': {'DEV': DATA_CACHED_DIR + 'develop.data', 'TRAIN': DATA_CACHED_DIR + 'training.data', 'TEST': DATA_CACHED_DIR + 'randomed_labeled_testing.data', 'NAH': ''},
    #                           }
    MODEL_WEIGHTS_DIR = DATA_DIR + 'my_model_weights.h5'
    PREDICTED_SCORE_DIR = DATA_DIR + 'predicted.score'
    WV_FILE_SUFFIX = 'vec.db'
    EXT_TOOL_DIR = '../ext_tool/'
    LOG_DIR = '../log/'
    LOG_FILE = LOG_DIR + 'log.txt'
    W2V_DIR = EXT_TOOL_DIR + 'word2vec-master/'
    W2V_EXE_DIR = W2V_DIR + 'word2vec.exe'
    W2V_RES_DIR = W2V_DIR + 'vectors.bin'
    PRETRAINED_VEC_DIR = DATA_CACHED_DIR + 'pretrained.vec'
    STOP_WORD_DIR = DATA_RAW_DIR + 'stop_words'

class PreProcessConfig(object):
    '''
     pre-processing in big picture includes 3 stages:
     cleaning text for vocab building
     vocab building
     cleaning text for qa data set
    '''
    CORPUS_MODE = '_PRETRAINED' # 'ENG_TEST', 'CHN_TEST', 'MIXED_CHN', '_PRETRAINED'...
    LING_UNIT = "WORD" # "WORD" and "CHAR" # must be "WORD" for "_PRETRAINED" corpus mode
    # text cleaning
    PUNCTUALATION_REMOVAL = True 
    STOP_WORD_REMOVAL = False 
    NUMBER_REMOVAL = True  
    # w2v training  
    # for WORD_DIM see ModelConfig
    W2V_ALGORITHM_CBOW = 0 # 1 for C-BOW, 0 for skip-gram
    W2V_WIN_SIZE = 8
    W2V_ITER = 15
    W2V_NEG_SAMP = 0
    W2V_HIER_SFTMX = 1
    # wv feature
    TO_NORM = True

class ModelConfig(object):
    SUPPORTED_DATASET = set(['HITNLP', 'TREC'])
    WORD_DIM_DICT = {'HITNLP': 300, 'ENG_TEST': 200, 'CHN_TEST': 200}
    
    SORT_INSTANCE = False
    PAD_WIDE = True
    # conv network config
    FEATURE_MAP_NUM_1 = 50
    CONV_FILTER_LEN_1 = 1
    MAX_SENT_LEN_1 = 300
    FEATURE_MAP_NUM_2 = 300
    CONV_FILTER_LEN_2 = 1
    MAX_SENT_LEN_2 = 300
    # train config
    BATCH_SIZE = 32
    TRAIN_EPOCH = 10
    LOSS_FUNC = 'binary_crossentropy'
    OPT = 'adadelta'
    