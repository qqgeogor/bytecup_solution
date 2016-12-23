import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder,StandardScaler,Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda,Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.utils.visualize_util import plot
import distance
import xgboost as xgb

seed = 1024
np.random.seed(seed)

path = "../"


def str_jaccard(str1, str2):
    res = distance.jaccard(str1, str2)
    return res


question_numeric = ['char_4_q','char_5_q','char_6_q']

train = pd.read_csv(path+'invited_info_train.txt',dtype={"expert_id":str,'question_id':str})
expert_id = train['expert_id'].values
expert_id = LabelEncoder().fit_transform(expert_id)

test = pd.read_csv(path+'test_nolabel.txt',dtype={"expert_id":str,'question_id':str}).fillna(-1)
test.columns = ['question_id','expert_id','label']
len_train = train.shape[0]
y = train['label'].values


mf_clf = [
        'lr',
        'lsvc',
        'ftrl',
        'fm',
        'nn',
        'gblinear',
        'gbtree',
        'rf',
        # 'et_2',
        'libfm',
        # 'cnn',
        'lr_desc',
        'lsvc_desc',
    ]


X= []
X_t = []
for f in mf_clf:
    mf = pd.read_pickle(path+'mf_%s_clf.pkl'%f)
    X_mf = mf[0]
    X_t_mf = mf[1]
    X_mf = np.expand_dims(X_mf,1)
    X_t_mf = np.expand_dims(X_t_mf,1)
    X.append(X_mf)
    X_t.append(X_t_mf)
X_emb = pd.read_pickle(path+'X_emb.pkl')
X_t_emb = pd.read_pickle(path+'X_t_emb.pkl')

X_svd = pd.read_pickle(path+'new_X_svd.pkl')
X_t_svd = pd.read_pickle(path+'new_X_t_svd.pkl')

X_nmf = pd.read_pickle(path+'new_X_nmf.pkl')
X_t_nmf = pd.read_pickle(path+'new_X_t_nmf.pkl')

X_tsne = pd.read_pickle(path+'new_X_tsne.pkl')
X_t_tsne = pd.read_pickle(path+'new_X_t_tsne.pkl')

X_svd_200 = pd.read_pickle(path+'new_X_svd_200.pkl')
X_t_svd_200 = pd.read_pickle(path+'new_X_t_svd_200.pkl')

X.append(X_emb)
X_t.append(X_t_emb)

X.append(X_nmf)
X_t.append(X_t_nmf)

X.append(X_svd)
X_t.append(X_t_svd)

X.append(X_svd_200)
X_t.append(X_t_svd_200)

X.append(X_tsne)
X_t.append(X_t_tsne)

X = np.hstack(X)
X_t = np.hstack(X_t)



skf = KFold(len(y), n_folds=10, shuffle=True, random_state=seed)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]
    train_test = train.iloc[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

print('X_train',X_train.shape,'X_test',X_test.shape)

X_train = X
y_train = y

from ndcg_code import ndcg_at_k
def ndcg_score(train_test,y_preds):
    train_test['predict'] = y_preds
    g = train_test.groupby(['question_id'])
    def p(x,k=5):
        x = x.sort_index(by='predict',ascending=False)
        r = x['label']
        return ndcg_at_k(r,k)
    
    res1 = g.apply(lambda x:p(x,k=5))

    res2 = g.apply(lambda x:p(x,k=10))

    return (res1.mean()+res2.mean())/2


'''
gbdt
'''
gbtree = xgb.XGBClassifier(n_estimators=300,learning_rate=0.025 ,max_depth=10,colsample_bytree=0.8,subsample=0.9,gamma=0.2,seed=seed)

gbtree.fit(
    X_train,
    y_train,
    eval_metric='logloss',
    eval_set=[(X_train,y_train),(X_test,y_test)],
    early_stopping_rounds=50,
    )
gbtree_y_preds = gbtree.predict_proba(X_test)[:,1]
score = ndcg_score(train_test,gbtree_y_preds)
print score
score = log_loss(y_test,gbtree_y_preds)
print score
score = roc_auc_score(y_test,gbtree_y_preds)
print score
gbtree_y_t_preds = gbtree.predict_proba(X_t)[:,1]
submission = pd.DataFrame()
submission['qid'] = test['question_id']
submission['uid'] = test['expert_id']
submission['label'] = gbtree_y_t_preds
submission.to_csv('submission_gbtree_stack.csv',index=False)
