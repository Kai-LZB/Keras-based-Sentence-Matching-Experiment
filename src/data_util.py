#! -*- coding:utf-8 -*-
'''

Data processing for model

@author: LouisZBravo

'''
import numpy as np
import config as cfg
import sqlite3
import re
from math import log

import sys
import os
ext_tool_dir = cfg.DirConfig.EXT_TOOL_DIR
sys.path.append(os.path.abspath(ext_tool_dir))
# ignore error msg here as long as jieba package in the right place
import jieba

digit_ch_set = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '０', '１', '２', '３', '４', '５', '６', '７', '８', '９'])
punc_re_str = "[][！？。｡＂＃＄％＆＇―（）—＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞·〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#%&\'()+,-/:;<=>@\\\^_`{}~\s\t\.\^\$\*\+\?\|]+"
class TextCleaner(object):
    '''
     text cleaner instance
     clean raw text for both vocab and sentence match model
     a full process includes segmentation(tokenizing) stop-word removal, number and proper noun processing
    '''
    
    '''
     clean the document provided
     a full process includes segmentation(tokenizing) stop-word removal, number and proper noun processing
     for data streaming use
     return a piece of cleaned text
    '''
    # for w2v training & model
    # save both to cached & w2v dir
    
    '''
     for english, we need to lowercase all letters and lemmatize words
    '''
    
    def __init__(self, text_path, punc_rmvl = None, s_w_rmvl = None, num_rmvl = None):
        self.text_path = text_path
        self.ling_unit = cfg.PreProcessConfig.LING_UNIT
        assert self.ling_unit in ["WORD", "CHAR"]
        if punc_rmvl is None:
            self.punc_rmvl = cfg.PreProcessConfig.PUNCTUALATION_REMOVAL
        else:
            self.punc_rmvl = punc_rmvl
        if s_w_rmvl is None:
            self.s_w_rmvl = cfg.PreProcessConfig.STOP_WORD_REMOVAL
        else:
            self.s_w_rmvl = s_w_rmvl
        if num_rmvl is None:
            self.num_rmvl = cfg.PreProcessConfig.NUMBER_REMOVAL
        else:
            self.num_rmvl = num_rmvl
        
    def clean_chn_line(self, line_raw_utf8, stop_set):
        '''
         given a string of Chinese
         return a list of clean version of the string
        '''
        line = line_raw_utf8
        line_seg = []
        line_seg_stop_w = []
        
        # replace punctuation with spacing to make separate items remain separated
        if self.punc_rmvl:
            line = re.sub(punc_re_str, " ", line)
        
        # further cleaning including segmentation, number removal
        if self.ling_unit == "WORD": # seg the text by word
            for w in jieba.cut(line):
                if(w != " " and w not in stop_set): # non-stop version
                    # make numbers zeroes
                    if self.num_rmvl:
                        word = self._remove_digit(w)
                    else:
                        word = w
                    line_seg.append(word)
                    line_seg_stop_w.append(word)
                elif w != " ": # stop word version
                    if self.num_rmvl:
                        word = self._remove_digit(w)
                    else:
                        word = w
                    line_seg_stop_w.append(word)
        else: # CHAR as linguistic unit
            for ch in line:
                if(ch != " " and ch not in stop_set): # non-stop version
                    # make numbers zeroes
                    if self.num_rmvl:
                        char_ = self._remove_digit(ch)
                    else:
                        char_ = ch
                    line_seg.append(char_)
                    line_seg_stop_w.append(char_)
                elif ch != " ": # stop word version
                    if self.num_rmvl:
                        char_ = self._remove_digit(ch)
                    else:
                        char_ = ch
                    line_seg_stop_w.append(char_)
                
        return line_seg, line_seg_stop_w
    
    def clean_chn_line_for_vocab_set(self, line_raw_utf8):
        '''
         given a string of Chinese
         return a set of all possible words of the string
        '''
        line = line_raw_utf8
        
        ret = set(jieba.lcut(line, cut_all=True))
        return ret
            
    
    def clean_chn_corpus_2file(self, save_path):
        f = open(self.text_path, 'rb') # use 'rb' mode for windows decode problem
        f_w = open(save_path, 'wb')
        # start to clean each line
        
        for l in f:
            line = l.decode("utf-8")
            
            line_seg = (self.clean_chn_line(line, set([])))[1]
            
            if len(line_seg) == 0:
                continue
            
            f_w.write(' '.join(line_seg).encode("utf-8"))
            f_w.write(' '.encode("utf-8"))
            
        f.close()
        f_w.close()
        
    def _remove_digit(self, s):
        flag_all_digit = True
        for ch in s:
            if(ch not in digit_ch_set): #< '0' or ch > '9'):
                flag_all_digit = False
                break
        if flag_all_digit:
            return '0'
        else:
            return s

class Vocab(object):
    '''
     Vocabulary instance for model
     
     Important attributes:
     word_matrix: the word matrix composed by concatenating words' distributional representation vectors
     word2id: map from word to id in the word matrix
     id2word: map from id to word in the word matrix
     word_dim: dimensionality of a word vector, number of columns in word matrix
     vocab_size: number of known words, number of rows in word matrix
     stop_set: a set of stop words
     __unk_mapping: map of OOV words to known words
     
     Will expand to cover all vocabulary appearing in text
    '''    
    def __init__(self, wv_path):
        self.word2idx = {}
        self.idx2word = {}
        self.wdim = 0
        self.vocab_size = 0
        self.wv_path = wv_path
        self.word_matrix = None
        self.stop_set = self.load_stop_word_set()
        self.__unk_mapping = {}
        self._unk_num = 0
        self.kn_set = set([])
        self.task_vocab_set = set([]) # for large vector source reading only
        
    
    def build_vocab_database(self, qa_data_mode, clean_corpus_filename):
        '''
         train word vectors on corpus and save them into database
         WORD_VECTORS:
         WORD TEXT | DIM0 REAL | DIM1 REAL | ... | DIMn REAL
         
         VECTOR_DIMENSIONALITY:
         WDIM
         
         VOCABULARY_SIZE(invisible):
         VSIZE
         
         UNKNOWN_WORD_MAPPING:
         WORD | SIMILAR_WORD
        '''
        
        wdim = cfg.ModelConfig.WORD_DIM_DICT[qa_data_mode]
        cbow = cfg.PreProcessConfig.W2V_ALGORITHM_CBOW
        win_size = cfg.PreProcessConfig.W2V_WIN_SIZE
        neg_samp = cfg.PreProcessConfig.W2V_NEG_SAMP
        hs = cfg.PreProcessConfig.W2V_HIER_SFTMX
        iter_ = cfg.PreProcessConfig.W2V_ITER
        
        #execute word2vec program
        print('Please launch the word2vec program')
        #write_log('Please launch the word2vec program')
        print('using parameters:')
        #write_log('using parameters:')
        print('-train %s -output vectors.bin -cbow %d -size %d -window %d -negative %d -hs %d -sample 1e-4 -threads 20 -binary 0 -iter %d\n' % (
            clean_corpus_filename, cbow, wdim, win_size, neg_samp, hs, iter_))
        #write_log('-train %s -output vectors.bin -cbow %d -size %d -window %d -negative %d -hs %d -sample 1e-4 -threads 20 -binary 0 -iter %d\n' % (clean_corpus_filename, cbow, wdim, win_size, neg_samp, hs, iter_))
        print('Press enter after word2vec program successfully generates vectors binary file "vecters.bin".\n')
        input()
        
        print ('Start reading vector data from word2vec program...')
        #write_log('Start reading vector data from word2vec program...')
        
        w2v_res_path = cfg.DirConfig.W2V_RES_DIR
        vec_bin_file = open(w2v_res_path, 'rb')
        line = vec_bin_file.readline()
        vsize = int(line.split()[0])
        wdim_read = int(line.split()[1])
        assert wdim == wdim_read
        
        
        widx = 0
        vector_list = []
        
        for l in vec_bin_file:
            line = l.decode("utf-8")
            if len(line.split()) != wdim + 1:
                write_log("length of line in vector file is not consistent with expected:")
                write_log("%s\n" % line)
                write_log("expected %d, but read %d" % (wdim, len(line.split())-1))
                continue
            word = line.split()[0]
            vector_b = line.split()[1:]
            vector = [float(i) for i in vector_b]
            vector_list.append(vector)
            self.word2idx[word] = widx
            self.idx2word[widx] = word
            widx += 1
            
        self.vocab_size = widx
        self.wdim = wdim_read
        # assert self.vocab_size == widx # unread empty word might result in difference in vsize stored and vsize read
        vec_bin_file.close()
        
        # initialize a empty word matrix 
        # zero vector is the last row
        self.word_matrix = np.zeros((self.vocab_size + 1, self.wdim), 
                                    dtype = np.float32
                                    )
        for cur_idx in range(self.vocab_size):
            self.word_matrix[cur_idx] = vector_list[cur_idx] # each row is a word vector
            
        print('Word vectors successfully loaded. Now saving to database...')
        
        try:
            os.remove(self.wv_path)
        except FileNotFoundError:
            pass
        
        
        conn = sqlite3.connect(self.wv_path)
        params = {}
        params["wdim"] = self.wdim
        params["vector_list"] = vector_list
        params["vsize"] = self.vocab_size
        self.create_tables_in_db(conn, params)
        conn.close()
        
    def build_vocab_db_from_pretrained(self, qa_data_mode, task_data_list):
        '''
         load word vectors from huge pre-trained source
         only save vectors needed by task data into database
         in 'WORD' only!!!
         
         WORD_VECTORS:
         WORD TEXT | DIM0 REAL | DIM1 REAL | ... | DIMn REAL
         
         VECTOR_DIMENSIONALITY:
         WDIM
         
         VOCABULARY_SIZE(invisible):
         VSIZE
         
         UNKNOWN_WORD_MAPPING:
         WORD | SIMILAR_WORD
         
        '''
        # parameters loading
        wdim = cfg.ModelConfig.WORD_DIM_DICT[qa_data_mode]
        wv_source_path = cfg.DirConfig.PRETRAINED_VEC_DIR
        
        
        
        if qa_data_mode == 'HITNLP':
            possible_word_set = set([])
            vec_bin_file = open(wv_source_path, 'rb') # read it through to extract new word for word segmentation dict
            line = vec_bin_file.readline()
            wdim_read = int(line.split()[1])
            for l in vec_bin_file:
                line = l.decode("utf-8")
                if len(line.split()) != wdim_read + 1:
                    continue
                word = line.split()[0]
                possible_word_set.add(word)
            vec_bin_file.close()
            for word in possible_word_set:
                jieba.add_word(word) # add possible new words to jieba dict
            self.task_vocab_set = self._load_wset_from_HITNLP_data(task_data_list)
        else:
            self.task_vocab_set = None
            write_log("No task dataset was provided for word to be loaded. Will use all words in vec file.")
            print("No task dataset was provided for word to be loaded. Will use all words in vec file.")
        
        vec_bin_file = open(wv_source_path, 'rb')
        line = vec_bin_file.readline()
        #vsize = int(line.split()[0]) #vsize in source vec file is not needed
        wdim_read = int(line.split()[1])
        assert wdim == wdim_read
        
        widx = 0
        vector_list = []
        
        for l in vec_bin_file:
            line = l.decode("utf-8")
            if len(line.split()) != wdim + 1:
                write_log("length of line in vector file is not consistent with expected:")
                write_log("%s\n" % line)
                write_log("expected %d, but read %d" % (wdim, len(line.split())-1))
                continue
            word = line.split()[0]
            vector_b = line.split()[1:]
            if word not in self.task_vocab_set:
                continue
            vector = [float(i) for i in vector_b]
            vector_list.append(vector)
            self.word2idx[word] = widx
            self.idx2word[widx] = word
            widx += 1
            
        write_log("%d words loaded." % widx)
        self.vocab_size = widx
        self.wdim = wdim_read
        
        vec_bin_file.close()
        
        # initialize a empty word matrix 
        # zero vector is the last row
        self.word_matrix = np.zeros((self.vocab_size + 1, self.wdim), 
                                    dtype = np.float32
                                    )
        for cur_idx in range(self.vocab_size):
            self.word_matrix[cur_idx] = vector_list[cur_idx] # each row is a word vector
            
        print('Word vectors successfully loaded. Now saving to database...')
        
        try:
            os.remove(self.wv_path)
        except FileNotFoundError:
            pass
        
        
        conn = sqlite3.connect(self.wv_path)
        params = {}
        params["wdim"] = self.wdim
        params["vector_list"] = vector_list
        params["vsize"] = self.vocab_size
        self.create_tables_in_db(conn, params)
        conn.close()
        
    
    def create_tables_in_db(self, conn, params):
        wdim = params["wdim"]
        vector_list = params["vector_list"]
        vsize = params["vsize"]
        
        c = conn.cursor()
        # create table for word vectors
        dim_col_strs = "" # string of column names to be filled into command
        for i in range(wdim):
            dim_col_strs = dim_col_strs + ", DIM%d REAL" % i #, dim0 real, dim1 real...
        # CREATE TABLE word_vecs (word text, dim0 real, ..., dimn real)
        crt_tbl_command = "CREATE TABLE WORD_VECTORS (WORD TEXT" + dim_col_strs + ")"
        c.execute(crt_tbl_command)
        
        # insert vector values into database
        q_marks = ""
        for i in range(self.wdim):
            q_marks = q_marks = q_marks + ", ?"
            
        ist_val_command = "INSERT INTO WORD_VECTORS VALUES (?" + q_marks + ")"
        wvec2db_lst = []
        for w_idx in range(self.vocab_size):
            wvec2db = [self.idx2word[w_idx]] # ['word']
            wvec2db.extend(vector_list[w_idx]) # ['word', dim1, dim2...]
            wvec2db_lst.append(tuple(wvec2db))
        
        c.executemany(ist_val_command, wvec2db_lst)
        conn.commit()
        
        # record vector dimensionality & vocab size
        crt_tbl_command = "CREATE TABLE VECTOR_DIMENSIONALITY (WDIM INTEGER)"
        c.execute(crt_tbl_command)
        crt_tbl_command = "CREATE TABLE VOCABULARY_SIZE (VSIZE INTEGER)"
        c.execute(crt_tbl_command)
        ist_val_command = "INSERT INTO VECTOR_DIMENSIONALITY VALUES (?)"
        c.execute(ist_val_command, (wdim,))
        ist_val_command = "INSERT INTO VOCABULARY_SIZE VALUES (?)"
        c.execute(ist_val_command, (vsize,))
        
        # create unknown word mapping
        crt_tbl_command = "CREATE TABLE UNKNOWN_WORD_MAPPING (WORD TEXT, SIMILAR_WORD TEXT)"
        c.execute(crt_tbl_command)
        
        conn.commit()
        
    def _load_wset_from_HITNLP_data(self, task_data_list):
        '''
         a HITNLP task specified method for loading word collection in task
        '''
        wset = set([])
        for task_data_file in task_data_list:
            text_cleaner = TextCleaner(task_data_file, punc_rmvl=False, s_w_rmvl=False, num_rmvl=False)
            f = open(task_data_file, 'rb')
            for l in f:
                line = l.decode('utf-8')
                line_item = line.split('\t')
                q_raw = line_item[0]
                a_raw = line_item[1]
                q_wset = text_cleaner.clean_chn_line_for_vocab_set(q_raw)
                a_wset = text_cleaner.clean_chn_line_for_vocab_set(a_raw)
                wset = wset | q_wset | a_wset
            f.close()
        return wset
    
    def load_wv_from_db(self, qa_data_mode):
        '''
         Load pre-trained word vectors from database
         WORD_VECTORS:
         WORD TEXT | DIM0 REAL | DIM1 REAL | ... | DIMn REAL
         
         VECTOR_DIMENSIONALITY:
         WDIM
         
         VOCABULARY_SIZE(invisible):
         VSIZE
         
         UNKNOWN_WORD_MAPPING:
         WORD | SIMILAR_WORD
        '''
        
        # get parameters on word representations
        to_norm  = cfg.PreProcessConfig.TO_NORM
        
        wv_path = self.wv_path
        conn = sqlite3.connect(wv_path)
        c = conn.cursor()
        
        # get word dimensionality
        c.execute("SELECT * FROM VECTOR_DIMENSIONALITY")
        (wdim_read, ) = c.fetchone()
        assert c.fetchone() == None
        wdim = cfg.ModelConfig.WORD_DIM_DICT[qa_data_mode]
        assert wdim == wdim_read
        self.wdim = wdim_read
        
        # get vocab size
        c.execute("SELECT * FROM VOCABULARY_SIZE")
        (vsize, ) = c.fetchone()
        assert c.fetchone() == None
        self.vocab_size = vsize
        
        # get word, word vectors and count their indices
        widx = 0
        vector_list = []
        exec_dict = {True: lambda vec: np.divide(vec, np.linalg.norm(vec)), False: lambda vec: vec} #dict with function addr
        for row in c.execute("SELECT * FROM word_vectors"):
            word = row[0] # row[0].decode('utf-8')
            vector = row[1:]
            vector_normed = exec_dict[to_norm](vector) # using selection dict w/ func addr instead of condition branch to save time
            vector_list.append(vector_normed)
            self.word2idx[word] = widx
            self.idx2word[widx] = word
            widx += 1
        assert widx == self.vocab_size
            
        # get oov word mapping
        for oov_word, sim_word in c.execute("SELECT * FROM UNKNOWN_WORD_MAPPING"):
            self.__unk_mapping[oov_word] = sim_word
            
        conn.close()
        
        assert self.vocab_size == widx
        
        # initialize a empty word matrix 
        # zero vector is the last row
        self.word_matrix = np.zeros((self.vocab_size + 1, self.wdim), 
                                    dtype = np.float32
                                    )
        for cur_idx in range(self.vocab_size):
            self.word_matrix[cur_idx] = vector_list[cur_idx] # each row is a word vector
            
            '''
            if cur_idx % 20 == 0:
                print("%d: %s" % (cur_idx, self.idx2word[cur_idx].decode("utf-8")))
                print(self.word_matrix[cur_idx])
            '''
    
    def has_word(self, word):
        ret = word in self.word2idx
        return ret
    
    def get_vector_by_index(self, index):
        return self.word_matrix[index]
    
    def get_um(self):
        # for testing
        conn = sqlite3.connect(self.wv_path)
        c = conn.cursor()
        cmd = "SELECT * FROM UNKNOWN_WORD_MAPPING"
        c.execute(cmd)
        db_ = c.fetchall() # from db
        ret = (db_, self.__unk_mapping)
        return ret
    
    def unk_map(self, oov_word):
        '''
         need to be re-wrote
         unknown words are generated during text cleaning of sentence pair file
        '''
        if oov_word in self.__unk_mapping:
            sim_word = self.__unk_mapping[oov_word]
        else: # not recorded
            # randomly map this word to a similar word
            self._unk_num += 1 ######################################################
            sim_idx = np.random.randint(0, self.vocab_size)
            sim_word = self.idx2word[sim_idx]
            # store oov word mapping
            self.__unk_mapping[oov_word] = sim_word
            '''
            # save mapping res in db
            conn = sqlite3.connect(self.wv_path)
            c = conn.cursor()
            ist_val_command = "INSERT INTO UNKNOWN_WORD_MAPPING VALUES (?, ?)"
            c.execute(ist_val_command, (oov_word, sim_word))
            conn.commit()
            conn.close()
            '''
        return sim_word
    
    def save_unk_word_2_db(self):
        '''
         should be called every time all data streams being generated
        '''
        conn = sqlite3.connect(self.wv_path)
        conn.close()
    
    def load_stop_word_set(self):
        stop_set = set([])
        s_w_dir = cfg.DirConfig.STOP_WORD_DIR
        try:
            f = open(s_w_dir, 'rb')
        except FileNotFoundError:
            write_log("Stop word file not found while initializing vocab before training word vectors.\n")
        else: 
            # every line is a stop word
            for l in f:
                line = l.decode("utf-8")
                s_w = line.strip('\n')[0]
                stop_set.add(s_w)
            f.close()
        return stop_set
    
    def get_stop_word_set(self):
        return self.stop_set
            
    def to_idx_sequence(self, sentence):
        '''
         transfer a sentence(list of word) to a sequence of indices
         linguistic unit in sentence separated by spacing
         specific condition handling strategy as follows:
         out-of-vocab word:
             seen as a random invariant idx
             db <- mem <- new found OOV
         vocab cleaning:
             ...
         stop word:
             3 approaches:
             using their original vector representations
             regarding them as OOV words
             totally ignore them
        '''
        idx_sequence = []
        for word in sentence:
            if self.has_word(word): # in-vocab word
                self.kn_set.add(word) ################################
                idx = self.word2idx[word]
            else: # OOV word
                sim_word = self.unk_map(word)
                idx = self.word2idx[sim_word]
                
                
            idx_sequence.append(idx)
        
        return idx_sequence
    def get_word_dimensionality(self):
        return self.wdim
    
    def get_zero_vec_idx(self):
        # in vector matrix the last row was set to zeroes
        # the index of the vsize+1th row is vsize
        return self.vocab_size
    

class SentenceDataStream(object):
    '''
     data stream of sentence pairs, made into batches
     words in sentence are stored as indices by the Vocab instance
     sentence pairs are sorted by sentence length
     important attributes:
         instance: list of all sentence pairs, sorted as above
         -----remember to make space for preprocess customization set by config parameters-----
         batches: iterator? of data to be sent into the model
         batch_span: list of tuples pointing positions of each batch in instance list
    '''
    def __init__(self, qa_file_path, vocab, mode):
        '''
         read question answer pair file and save qa pairs into memory
         mode: a tuple of dataset name and train/evaluation mode, e.g. ('HITNLP', 't')
        '''
        
        self.qa_data_mode = mode[0]
        self.t_e_mode = mode[1]
        assert self.qa_data_mode in cfg.ModelConfig.SUPPORTED_DATASET
        assert self.t_e_mode in ('t', 'e')
        self.vocab = vocab
        self.text_cleaner = TextCleaner(qa_file_path)
        
        self.instances = [] # each instance consists of word idx seqs of q, a and label if in 't' mode 
        self.instance_size = 0
        self.batch_size = cfg.ModelConfig.BATCH_SIZE
        self.batch_span = [] # tuples recording start and end index in instance of each batch
        self.q_idx_matrix_batches = [] # matrix consists of padded sequence of indices
        self.a_idx_matrix_batches = []
        self.label_batches = []
        self.add_feat_batches = []
        self._df_dict = {}
        
        if self.qa_data_mode == 'HITNLP':
            self._prepare_HITNLP_data(qa_file_path)
        else:
            pass
        
    def _prepare_HITNLP_data(self, qa_file_path):
        '''
         make data stream based on qa file
         this process first reads instances line by line and clean them
         then the text are translated to the form of word indices
         so that the data set is ready to generate data stream to the model
        '''
        s_w_rmvl = cfg.PreProcessConfig.STOP_WORD_REMOVAL
        stop_set = self.vocab.get_stop_word_set()
            
        to_sort = cfg.ModelConfig.SORT_INSTANCE
        max_sent_len = cfg.ModelConfig.MAX_SENT_LEN
        
        f = (open(qa_file_path, 'rb'))
        for l in f:
            line = l.decode('utf-8')
            line_item = line.split('\t')
            q_raw = line_item[0]
            a_raw = line_item[1]
            if self.t_e_mode == 't':
                label = int(line_item[2])
            # clean qa pair sentence
            q_seg_non_s, q_seg_s_w = self.text_cleaner.clean_chn_line(q_raw, stop_set)
            a_seg_non_s, a_seg_s_w = self.text_cleaner.clean_chn_line(a_raw, stop_set)
            # translate word to word indices
            if s_w_rmvl:
                q_idx_seq = self.vocab.to_idx_sequence(q_seg_non_s)
                a_idx_seq = self.vocab.to_idx_sequence(a_seg_non_s)
            else:
                q_idx_seq = self.vocab.to_idx_sequence(q_seg_s_w)
                a_idx_seq = self.vocab.to_idx_sequence(a_seg_s_w)
            # calculate statistical evaluation basis: w_overlap, idf-weighed or not, stop_w or not
            w_overlap = self.calc_word_overlap(q_seg_s_w, a_seg_s_w)
            n_s_overlap = self.calc_word_overlap(q_seg_non_s, a_seg_non_s)
            idf_overlap = self.calc_idf_word_overlap(q_seg_s_w, a_seg_s_w)
            idf_n_s_overlap = self.calc_idf_word_overlap(q_seg_non_s, a_seg_non_s)
            
            if self.t_e_mode == 't':
                self.instances.append([q_idx_seq, a_idx_seq, label, w_overlap, n_s_overlap, idf_overlap, idf_n_s_overlap])
            else:
                self.instances.append([q_idx_seq, a_idx_seq, w_overlap, n_s_overlap, idf_overlap, idf_n_s_overlap]) # in evaluation mode results are generated by model and saved later
        
        self.instance_size = len(self.instances)
        
        # calculate all idf related score
        for i in range(self.instance_size):
            # idf_overlap
            idf_lst = self.instances[i][-2]
            calculated_idf = 0.0
            for idf_calc in idf_lst:
                calculated_idf += idf_calc.calc_idf(2 * self.instance_size)
            self.instances[i][-2] = calculated_idf
            # idf_n_s_overlap
            idf_n_s_lst = self.instances[i][-1]
            calculated_idf = 0.0
            for idf_calc in idf_n_s_lst:
                calculated_idf += idf_calc.calc_idf(2 * self.instance_size)
            self.instances[i][-1] = calculated_idf
            
        f.close()
        
        """check if index sequence of dataset is a correct match"""
        # sort instances
        if to_sort:
            self.instances = sorted(self.instances, key = lambda instances: (len(instances[0]), len(instances[1])))
        """ does variable length affects performance of convolution filter? """
        # make batch idx
        self.batch_span = self.make_patch_span(self.batch_size, self.instance_size)
        # make batch content in terms of word idx
        for (batch_start, batch_end) in self.batch_span:
            q_idx_seq_lst = []
            a_idx_seq_lst = []
            add_feat_lst = []
            if self.t_e_mode == 't':
                label_lst = []
            for i in range(batch_start, batch_end):
                q_idx_seq_lst.append(self.instances[i][0]) # element: a sequence of word indices
                a_idx_seq_lst.append(self.instances[i][1])
                add_feat_lst.append([self.instances[i][-4],
                                    self.instances[i][-3],
                                    self.instances[i][-2],
                                    self.instances[i][-1]]
                                    )
                if self.t_e_mode == 't':
                    label_lst.append(self.instances[i][2])
            # padding
            q_len_lst = [len(seq) for seq in q_idx_seq_lst]
            a_len_lst = [len(seq) for seq in a_idx_seq_lst]
            max_q_len = min(max_sent_len, max(q_len_lst))
            max_a_len = min(max_sent_len, max(a_len_lst))
            
            (padded_q_idx_seq_lst, p_q_s_len) = self._pad(q_idx_seq_lst, max_q_len)
            (padded_a_idx_seq_lst, p_a_s_len) = self._pad(a_idx_seq_lst, max_a_len)
            """check if the padded sequences have the same lenth, both for wide pad and not"""
            cur_batch_size = batch_end - batch_start
            q_batch = np.zeros((cur_batch_size, p_q_s_len),
                               dtype = np.int32
                               )
            a_batch = np.zeros((cur_batch_size, p_a_s_len),
                               dtype = np.int32
                               )
            add_feat_batch = np.zeros((cur_batch_size, 4),
                                      dtype = np.float32
                                      )
            for i in range(cur_batch_size):
                q_batch[i] = np.array(padded_q_idx_seq_lst[i], dtype=np.int32)
                a_batch[i] = np.array(padded_a_idx_seq_lst[i], dtype=np.int32)
                add_feat_batch[i] = np.array(add_feat_lst[i], dtype=np.float32)
            """check if padded sequences are converted to np matrix, both for wide pad and not"""
            self.q_idx_matrix_batches.append(q_batch)
            self.a_idx_matrix_batches.append(a_batch)
            self.add_feat_batches.append(add_feat_batch)
            if self.t_e_mode == 't':
                self.label_batches.append(np.array(label_lst, dtype = np.int32))
            
            """check the dimensionality of these matrices when network doesn't accept them"""
            # write_log("q_batch_size: " + str(q_batch.shape))
            # write_log("a_batch_size: " + str(a_batch.shape))
            
    def calc_word_overlap(self, q_seg, a_seg):
        '''take 2 lists of segmentation words and calculate word overlap measurement'''
        q_set = set(q_seg)
        a_set = set(a_seg)
        overlap_set = q_set & a_set
        overlap_num = len(overlap_set) 
        sum_set = q_set | a_set
        sum_num = len(sum_set)
        ret = float(overlap_num) / float(len(q_seg)) # dice for float(sum_num); overlap for float(len(q_seg))
        return ret
    
    def calc_idf_word_overlap(self, q_seg, a_seg):
        '''
         return a list of idf calculators
         each calculator calc 1 / |Q| * idf at the end of a data stream
         so in the end all we have to do is to sum up the calculated value for each instance
        '''
        q_set = set(q_seg)
        a_set = set(a_seg)
        overlap_set = q_set & a_set
        overlap_num = len(overlap_set) 
        sum_set = q_set | a_set
        sum_num = len(sum_set)
        res = []
        for w in overlap_set:
            idf_calculator = IDFCalculator(w, self._df_dict, float(overlap_num) / float(len(q_seg))) # dice for float(sum_num); overlap for float(len(q_seg))
            res.append(idf_calculator)
        
        # calc global df one by one qa pair
        for w in q_set:
            if w in self._df_dict:
                self._df_dict[w] += 1
            else:
                self._df_dict[w] = 1
        
        for w in a_set:
            if w in self._df_dict:
                self._df_dict[w] += 1
            else:
                self._df_dict[w] = 1
        
        return res
    def make_patch_span(self, batch_size, instance_size):
        '''record index of each batch'''
        batch_span = []
        batch_num = int(np.ceil(float(instance_size) / float(batch_size)))
        for i in range(batch_num):
            batch_span.append((i * batch_size, min(instance_size, (i+1)*batch_size)))
        return batch_span
    
    def get_batch(self):
        '''
         a generator generates data to feed into the model directly
         yields a tuple of tensors:
         (q_batch, a_batch, label_batch) in 't' mode
         (q_batch, a_batch) in 'e' mode
         a q_batch is 3 dimensional: (sentence, word, vector dim)
         same for a_batch
         a label batch is 1 dimensional: (sentence pair)
        '''
        batch_num = 0
        batch_span = self.batch_span
        # batch_size = self.batch_size
        wdim = self.vocab.get_word_dimensionality()

        for (batch_start, batch_end) in batch_span:
            q_idx_batch_matrix = self.q_idx_matrix_batches[batch_num]
            a_idx_batch_matrix = self.a_idx_matrix_batches[batch_num]
            # generate 3-dim tensors:(sentence, word, dim)
            # for question and answer batch
            q_mtx_batch_dim0 = q_idx_batch_matrix.shape[0]
            q_mtx_len_dim1 = q_idx_batch_matrix.shape[1]
            a_mtx_batch_dim0 = a_idx_batch_matrix.shape[0]
            a_mtx_len_dim1 = a_idx_batch_matrix.shape[1]
            q_batch_tensor = np.zeros((q_mtx_batch_dim0,
                                       q_mtx_len_dim1,
                                       wdim),
                                      dtype = np.float32
                                      )
            a_batch_tensor = np.zeros((a_mtx_batch_dim0,
                                       a_mtx_len_dim1,
                                       wdim),
                                      dtype = np.float32
                                      )
            for i in range(q_mtx_batch_dim0):
                for j in range(q_mtx_len_dim1):
                    q_batch_tensor[i][j] = self.vocab.get_vector_by_index(q_idx_batch_matrix[i][j])
            for i in range(a_mtx_batch_dim0):
                for j in range(a_mtx_len_dim1):
                    a_batch_tensor[i][j] = self.vocab.get_vector_by_index(a_idx_batch_matrix[i][j])
            
            if self.t_e_mode == 't':
                yield (q_batch_tensor, a_batch_tensor, self.label_batches[batch_num], self.add_feat_batches[batch_num])
            else:
                yield (q_batch_tensor, a_batch_tensor, self.add_feat_batches[batch_num])
            batch_num += 1
    
    def get_batch_size(self):
        return self.batch_size
    
    def _pad(self, idx_seq_lst, max_seq_len):
        '''
         pad sequences in the list to the max length
         return a tuple: (padded index-sequence list, index-sequence length)
        '''
        zero_vec_idx = self.vocab.get_zero_vec_idx()
        padded_idx_seq_lst = []
        padded_idx_seq_len = max_seq_len
        for idx_seq in idx_seq_lst:
            padded_idx_seq = idx_seq # redundant invalid copy?
            if len(idx_seq) < max_seq_len: # to pad
                padded_idx_seq = idx_seq + [zero_vec_idx for _ in range(max_seq_len-len(idx_seq))]
            elif len(idx_seq) > max_seq_len: # to truncate
                padded_idx_seq = idx_seq[:max_seq_len]
            assert len(padded_idx_seq) == padded_idx_seq_len
            padded_idx_seq_lst.append(padded_idx_seq)
        return (padded_idx_seq_lst, padded_idx_seq_len)

class IDFCalculator(object):
    '''a mutable object that is expected to compute idf value after going through the entire dataset'''
    def __init__(self, word, df_dict, mutiplied_value):
        self.word = word
        self.df_dict = df_dict
        self.mutiplied_value = mutiplied_value
        
    def calc_idf(self, d_all_num):
        res = float(d_all_num) / float(self.df_dict[self.word])
        res = self.mutiplied_value * log(res)
        return res
        

def write_log(log_str):
    log_dir = cfg.DirConfig.LOG_DIR
    log_file_path = cfg.DirConfig.LOG_FILE
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    f = open(log_file_path, 'a')
    f.write(log_str)
    f.write('\n')
    f.close()