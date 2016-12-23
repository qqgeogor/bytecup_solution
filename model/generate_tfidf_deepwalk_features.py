import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder,StandardScaler,Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import TruncatedSVD,NMF
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint,Callback
from keras.layers.noise import GaussianNoise
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.utils.visualize_util import plot
import distance
import xgboost as xgb

seed = 1024
np.random.seed(seed)


path = "../"

# path = "E:\\ByteCup\\"

def get_tfidf(data,question_tfidf,expert_tfidf):
    data_q = []
    data_e = []
    for d in data[features].values:
        q=d[0]
        e=d[1]

        q_emb = question_tfidf[e]
        e_emb = expert_tfidf[q]
        data_q.append(q_emb)
        data_e.append(e_emb)
    if ssp.issparse(question_tfidf):
        data_q = ssp.vstack(data_q)
        data_e = ssp.vstack(data_e)
        data = ssp.hstack([data_q,data_e])
    else:
        data_q = np.vstack(data_q)
        data_e = np.vstack(data_e)
        data = np.hstack([data_q,data_e])
        
    return data

def read_emb(path):
    count=0
    f = open(path,'r')
    emb_dict = dict()
    for line in f:
        if count==0:
            count+=1
            continue
        line = line.split(' ')
        id = int(line[0])
        
        weights = line[1:]
        weights = np.array([float(i) for i in weights])
        count+=1
        emb_dict[id] = weights
    return emb_dict

def generate_doc(df,name,concat_name):
    res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
    res.columns = [name,'%s_doc'%concat_name]
    return res

train = pd.read_csv(path+'invited_info_train.txt',dtype={"expert_id":str,'question_id':str})
expert_id = train['expert_id'].values

valid = pd.read_csv(path+'validate_nolabel.txt',dtype={"expert_id":str,'question_id':str}).fillna(-1)
valid.columns = ['question_id','expert_id','label']
len_valid = valid.shape[0]

test = pd.read_csv(path+'test_nolabel.txt',dtype={"expert_id":str,'question_id':str}).fillna(-1)
test.columns = ['question_id','expert_id','label']
len_train = train.shape[0]

data = pd.concat([train,valid,test])

le = LabelEncoder()
data['question_id'] = le.fit_transform(data['question_id'].values)
data['expert_id'] = le.fit_transform(data['expert_id'].values)


y = train['label'].values
features = [
    'question_id',
    'expert_id',
    ]

train = data[:len_train]
test = data[len_train+len_valid:]

num_q = len(np.unique(data.values[:,0]))
num_e = len(np.unique(data.values[:,1]))

question_doc = generate_doc(data,name='expert_id',concat_name='question_id')
expert_doc = generate_doc(data,name='question_id',concat_name='expert_id')


'''
Generate deepwalk features
'''
question_doc['question_id_doc'].astype(str).to_csv('question_doc.adjlist',index=False)
expert_doc['expert_id_doc'].astype(str).to_csv('expert_doc.adjlist',index=False)

import commands
commands.getoutput("bash train_deepwalk.sh")
def read_emb(path):
    count=0
    f = open(path,'r')
    emb_dict = dict()
    for line in f:
        if count==0:
            count+=1
            continue
        line = line.split(' ')
        id = int(line[0])
        
        weights = line[1:]
        weights = np.array([float(i) for i in weights])
        count+=1
        emb_dict[id] = weights
    return emb_dict

q_dict = read_emb('question.emb')
e_dict = read_emb('expert.emb')

data_emb = []
for d in data[features].values:
    q=d[0]
    e=d[1]
    if q in q_dict:
        q_emb = q_dict[q].tolist()
    else:
        q_emb = [0]*64
    if e in e_dict:
        e_emb = e_dict[e].tolist()
    else:
        e_emb = [0]*64
    data_emb.append(q_emb+e_emb)
#     break
data_emb = np.array(data_emb)
X = data_emb[:len_train]
X_t = data_emb[len_train+len_valid:]
pd.to_pickle(X,path+"X_emb.pkl")
pd.to_pickle(X_t,path+"X_t_emb.pkl")

print (X.shape,X_t.shape)

'''
Generate tfidf features
'''
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
tfidf_e = TfidfVectorizer(ngram_range=(1,1))
tfidf_q = TfidfVectorizer(ngram_range=(1,1))

question_doc['expert_id']=question_doc['expert_id'].astype(int)
question_doc = question_doc.sort_values('expert_id')

expert_doc['question_id']=expert_doc['question_id'].astype(int)
expert_doc = expert_doc.sort_values('question_id')

question_tfidf = tfidf_q.fit_transform(question_doc['question_id_doc'].values).tocsr()
expert_tfidf = tfidf_e.fit_transform(expert_doc['expert_id_doc'].values).tocsr()
pd.to_pickle(question_tfidf,path+'question_tfidf.pkl')
pd.to_pickle(expert_tfidf,path+'expert_tfidf.pkl')


train_sparse = get_tfidf(train,question_tfidf,expert_tfidf)
test_sparse = get_tfidf(test,question_tfidf,expert_tfidf)

onehot =OneHotEncoder()
onehot.fit(data[features].values)
train_oh = onehot.transform(train[features].values)
test_oh = onehot.transform(test[features].values)

new_X = ssp.hstack([
        train_oh,
        train_sparse,
    ])

new_X_t = ssp.hstack([
        test_oh,
        test_sparse,
    ])
pd.to_pickle(new_X,path+'new_X.pkl')
pd.to_pickle(new_X_t,path+'new_X_t.pkl')

print (new_X.shape,new_X_t.shape)



'''
Generate svd features
'''
question_svd = TruncatedSVD(n_components=8).fit_transform(question_tfidf)
expert_svd = TruncatedSVD(n_components=8).fit_transform(expert_tfidf)

train_svd = get_tfidf(train,question_svd,expert_svd)
test_svd = get_tfidf(test,question_svd,expert_svd)

pd.to_pickle(train_svd,path+'new_X_svd.pkl')
pd.to_pickle(test_svd,path+'new_X_t_svd.pkl')
print (train_svd.shape,test_svd.shape)


'''
Generate svd features
'''
question_svd = TruncatedSVD(n_components=200).fit_transform(question_tfidf)
expert_svd = TruncatedSVD(n_components=200).fit_transform(expert_tfidf)

train_svd = get_tfidf(train,question_svd,expert_svd)
test_svd = get_tfidf(test,question_svd,expert_svd)

pd.to_pickle(train_svd,path+'new_X_svd_200.pkl')
pd.to_pickle(test_svd,path+'new_X_t_svd_200.pkl')
print (train_svd.shape,test_svd.shape)

'''
Generate nmf features
'''
question_nmf = NMF(n_components=4).fit_transform(question_tfidf)
expert_nmf = NMF(n_components=4).fit_transform(expert_tfidf)

train_nmf = get_tfidf(train,question_nmf,expert_nmf)
test_nmf = get_tfidf(test,question_nmf,expert_nmf)

pd.to_pickle(train_nmf,path+'new_X_nmf.pkl')
pd.to_pickle(test_nmf,path+'new_X_t_nmf.pkl')
print (train_nmf.shape,test_nmf.shape)



'''
Generate tsne features
'''
question_svd = TruncatedSVD(n_components=200).fit_transform(question_tfidf)
expert_svd = TruncatedSVD(n_components=200).fit_transform(expert_tfidf)


from tsne import bh_sne
question_tsne = bh_sne(question_svd)
expert_tsne = bh_sne(expert_svd)

train_tsne = get_tfidf(train,question_tsne,expert_tsne)
test_tsne = get_tfidf(test,question_tsne,expert_tsne)

pd.to_pickle(train_tsne,path+'new_X_tsne.pkl')
pd.to_pickle(test_tsne,path+'new_X_t_tsne.pkl')
print (train_tsne.shape,test_tsne.shape)
