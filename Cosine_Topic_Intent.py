import pandas as pd
import numpy as np
import string
import pickle
import re
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import time
import shutil
import sys

lstage_prev = 'Stage1'
lstage = 'Stage2'
outputDir=''   #Output files
rootDir= '' #Saving the pickel file model wise
tempDir= '' #Temp the pickel file model wise
archiveDir = ''


model_name = 'Generic'
data = pd.read_excel(outputDir + lstage_prev + '_' + model_name +'_output.xlsx',encoding='latin-1')

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "custom_pickle"
        return super().find_class(module, name)
        
def loadTfidfFile(model_name,name):
    vec_loc = rootDir + model_name + '_vector' + '\\' + str(name) + ".json.gz_vector.pkl"
    with open(vec_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        vec = unpickler.load()
    return vec

def loadTfidfFileIntent(model_name,name):
    vec_loc = rootDir + model_name + '_vector_features' + '\\' + str(name) + ".json.gz_vector_features.pkl"
    with open(vec_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        vec = unpickler.load()
    return vec

def categories(model_name):
    path = rootDir + model_name + '_vector' +  '\\'
    file = glob.glob(path +'/*.json.gz_vector.pkl')
    category_names =  []
    for i in range(len(file)):
        basename = os.path.basename(file[i])
        category_names.append(basename.split('.json.gz_vector.pkl')[0])
    return category_names


def cosine_cal(data,model_name,lstage,outputDir):

    data['Topic'] = data['Topic'].astype('category')
    topic_names = data['Topic'].unique().tolist()
    category_names = categories(model_name)
    topic_intent = [(topic,intent) for topic in topic_names for intent in category_names]
    index = 0
    
    alist = []
    for topic,intent in topic_intent:
        data_topic = data.loc[data['Topic']==topic]
        COMMENT = 'Ticket_Description'
        vec = loadTfidfFile(model_name,intent)
        intent_vec = loadTfidfFileIntent(model_name,intent)
        topic_vec = vec.transform(data_topic[COMMENT]).toarray()
        cosine_similarities = cosine_similarity(topic_vec,intent_vec)
        similarity = sorted(cosine_similarities[0])
        alist.append([topic,intent,similarity[-1]])
        
    df = pd.DataFrame(alist,columns = ['Topic','Intent','Score'])
    res = df.pivot_table(index=['Topic'], columns='Intent', values='Score', aggfunc='first').reset_index()
    res['Confidence_level'] = res[category_names].max(axis=1)
    res['Pred_Intent'] = res[category_names].idxmax(axis=1)
    res = res[['Topic','Confidence_level','Pred_Intent']]
    data = pd.merge(data, res, on="Topic",how='left')
    data.to_excel(outputDir + lstage + '_' + model_name +'_output.xlsx',index = False)
    
def tokenize(s): 
    re_tok = re.compile(f'([{string.punctuation}“”¨_«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split() 
    
# topic_model(data, model_name,lstage,outputDir)

cosine_cal(data,model_name,lstage,outputDir)
    



    
