import pandas as pd, numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import time
import shutil
import sys
import glob
    
    
th = 0.5
lstage_prev = ''
lstage = ''

outputDir= ''   #Output files
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

def cosine_intent_cal(data,model_name,lstage,outputDir):
    
	data_th = data[data['Confidence_level'] < th]
	data_th['T'] = data_th['Topic'].astype('category')
	ticket_names = data_th['Topic'].unique().tolist()
    COMMENT = 'Ticket_Description'
    data_th[COMMENT].fillna("unknown", inplace=True)
    category_names = categories(model_name)
    tkt_intent = [(ticket,intent) for ticket in ticket_names for intent in category_names]

    alist=[]    
    for ticket,intent in tkt_intent :
	    data_tkt = data.loc[data['Ticket_No']==ticket]
        vec = loadTfidfFile(model_name,intent)
        intent_vec = loadTfidfFileIntent(model_name,intent)
        test_vec = vec.transform(data_tkt[COMMENT]).toarray()
        cosine_similarities = cosine_similarity(test_vec,intent_vec)
        similarity = sorted(cosine_similarities[0])                                 
        alist.append(ticket,intent,similarity[-1]])
                            
    df = pd.DataFrame(alist,columns = ['Ticket_No','Pred_Intent','Score'])
    res = df.pivot_table(index = 'Ticket_No' ,columns='Pred_Intent', values='Score', aggfunc='first').reset_index()
    res['Confidence_level'] = res[category_names].max(axis=1)
    res['Pred_Intent'] = res[category_names].idxmax(axis=1)
    res['Pred_Intent']= np.where(res['Confidence_level'] > th, res['Pred_Intent'] , 'Others')
    res = res[['Ticket_No','Confidence_level','Pred_Intent']]
    data.loc[data['Ticket_No'].isin(res['Ticket_No']), ['Confidence_level', 'Pred_Intent']] = res[['Confidence_level', 'Pred_Intent']].values
    data[['Level_1','Level_2']] = data.Pred_Intent.str.split("-",expand=True,)
    data.to_excel(outputDir + lstage + '_' + model_name +'_output.xlsx',index = False)

def tokenize(s): 
    re_tok = re.compile(f'([{string.punctuation}“”¨_«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split() 
    
cosine_cal(data,model_name,lstage,outputDir)
    