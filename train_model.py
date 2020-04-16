import os
import os.path
import re
import sys
import getopt
import string
import shutil
import tarfile
import pandas as pd, numpy as np
import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import gzip

def tokenize(s): 
    re_tok = re.compile(f'([{string.punctuation}“”¨_«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()  

def createModel(data, tempDir, model_name):
    data['Intent']= data['Intent'].astype('category')
    label_cols = [k for k in data['Intent'].value_counts().keys() if data['Intent'].value_counts()[k] > 5]

    for index,name in enumerate(label_cols):
        pData_intent = data.loc[data['Intent'] == name]
        COMMENT = 'Sample'
        pData_intent[COMMENT].fillna("unknown", inplace = True)
        vec = TfidfVectorizer(ngram_range=(1,5),
                              stop_words="english",
                              analyzer='char',
                              tokenizer=tokenize,
                              # min_df=3, max_df=0.9, 
                              strip_accents='unicode',
                              use_idf=1,
                              smooth_idf=1, 
                              sublinear_tf=1)
        print('Creating vector for intent: ', name)
        vec.fit(pData_intent[COMMENT])
        intent_vec = vec.transform(pData_intent[COMMENT])
        pFolderName = ['_Vector','_Vector_features']
        for foldername in pFolderName:
            if not os.path.exists(tempDir + '\\' +  str(model_name) + foldername):
                os.makedirs(tempDir + '\\' +   str(model_name) + foldername)
            moduleName = tempDir + '\\' +   str(model_name) + foldername
            print(moduleName)
            if foldername == '_Vector':
                vec_loc = tempDir + '\\' +  str(model_name) + foldername + '\\' + str(name).replace('/','or') + ".vector.pkl"
                pickle.dump(vec, open(vec_loc,'wb'))
            else:
                vec_loc = tempDir + '\\' +  str(model_name) + foldername + '\\' + str(name).replace('/','or') + ".vector_features.pkl" 
                pickle.dump(intent_vec, open(vec_loc,'wb'))
    return(0)
	
createModel(data, tempDir, model_name)
