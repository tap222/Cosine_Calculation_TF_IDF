import pickle
import pandas as pd
import re
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import time
import shutil
import sys
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))



def topic_num(pData):

    COMMENT = 'Ticket_Description'
    pData[COMMENT].fillna("unknown", inplace=True)
    word_tokens = word_tokenize(pData[COMMENT]) 
      
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
      
    filtered_sentence = [] 
      
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
            
    n_topics = int(len(pData)\len(filtered_sentence))
    
    return n_topics
    
lstage = 'Stage1'
data = pd.read_excel('',encoding='latin-1')
model_name = 'Generic'

outputDir=''   #Output files
rootDir= '' #Saving the pickel file model wise
tempDir= '' #Temp the pickel file model wise
archiveDir = ''

def topic_model(pData, pModelName, outputDir, n_topics):
    try:
        lstage = 'Stage1'
        COMMENT = 'Ticket_Description'
        pData[COMMENT].fillna("unknown", inplace=True)
        count_vect = CountVectorizer(analyzer='word',       
                                     min_df=1,   # minimum reqd occurences of a word 
                                     stop_words='english',  # remove stop words
                                     lowercase=True,        # convert all words to lowercase
                                     token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                     max_features=50000,  # max number of uniq words  
                                     max_df=0.9, 
                                     strip_accents='unicode'                         
                                    )

        doc_term_matrix = count_vect.fit_transform(pData[COMMENT].values.astype('U'))
        LDA = LatentDirichletAllocation(n_components=n_topics,               # Number of topics
                                              max_iter=10,                   # Max learning iterations
                                              learning_method='online',   
                                              random_state=100,               # Random state
                                              batch_size=128,                 # n docs in each learning iter
                                              evaluate_every = -1,            # compute perplexity every n iters, default: Don't
                                              n_jobs = -1                   # Use all available CPUs
                                             )
        LDA.fit(doc_term_matrix)
        topic_values = LDA.transform(doc_term_matrix)
        pData['Topic'] = topic_values.argmax(axis=1)
        pData['Topic'] = 'Topic_' + data['Topic'].astype(str)
        pData.to_excel(outputDir + lstage + '_' + model_name +'_output.xlsx',index = False)

    except Exception as e:
        print(e)
        print('*** ERROR[XXX]: Loading XLS - Could be due to using non-standard template ***', sys.exc_info()[0],str(e), ' Training Id ',str(pTrainingID))
        return(-1, pData)
       
    return(0,pData)

n_topics  = topic_num(pData)

topic_model(data, model_name, outputDir, n_topics)