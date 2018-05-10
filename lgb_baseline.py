# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import warnings
import sys
sys.path.append("/notebook/shared/extra/")
warnings.filterwarnings("ignore")
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer  
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

data_path = "/notebook/models/nuoyan/Tencent_data/preliminary_contest_data/"

ad_feature=pd.read_csv(data_path + 'adFeature.csv')
if os.path.exists(data_path + 'userFeature.csv'):
    user_feature=pd.read_csv(data_path + 'userFeature.csv')
else:
    userFeature_data = []
    with open(data_path + 'userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv(data_path + 'userFeature.csv', index=False)

train=pd.read_csv(data_path + 'train.csv')
predict=pd.read_csv(data_path + 'test1.csv')
print("train shape:", train.shape, "test shape:", predict.shape)
print("load data prepared!") 

train.loc[train['label']==-1,'label']=0

train=pd.merge(train,ad_feature,on='aid',how='left')
train=pd.merge(train,user_feature,on='uid',how='left')
train=train.fillna('-1')

predict['label']=-1
predict=pd.merge(predict,ad_feature,on='aid',how='left')
predict=pd.merge(predict,user_feature,on='uid',how='left')
predict=predict.fillna('-1')

import numpy
import random
import pandas as pd
import scipy.special as special
import math
from math import log

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        #产生样例数据
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        #更新策略
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        #迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)
    

print(train.shape, predict.shape)

def nlp_feature_score(feature, train, predict):
    feature_count = {}
    feature_label_count = {}
    feature_list = train[feature].values.tolist()
    label_list = train['label'].tolist()
    for i in range(train.shape[0]):
        for item in feature_list[i].split(" "):
            if item in feature_count.keys():
                feature_count[item] += 1
            else:
                feature_count[item] = 1
            if (item not in feature_label_count.keys()) and (label_list[i]==1):
                feature_label_count[item] = 1
            elif (item in feature_label_count.keys()) and (label_list[i]==1):
                feature_label_count[item] += 1
    dianji=[]
    zhuanhua = []
    for item in feature_label_count.keys():
        dianji.append(feature_count[item])
        zhuanhua.append(feature_label_count[item])
    c = pd.DataFrame({'dianji':dianji,'zhuanhua':zhuanhua})
    hyper = HyperParam(1, 1)
    hyper.update_from_data_by_FPI(c['dianji'], c['zhuanhua'], 1000, 0.00000001)
    train_feature_score = []
    for i in range(train.shape[0]):
        score = 1
        for item in feature_list[i].split(" "):
            if item not in feature_label_count.keys():
                score += 0
            else:
                score  = score *(1- (feature_label_count[item]+hyper.alpha)/(feature_count[item]+hyper.alpha+hyper.beta))
        train_feature_score.append(score)
    test_feature_score = []
    test_feature_list = predict[feature].values.tolist()
    for i in range(predict.shape[0]):
        score = 1
        for item in test_feature_list[i].split(" "):
            if item not in feature_label_count.keys():
                score += 0
            else:
                score  = score *(1- (feature_label_count[item]+hyper.alpha)/(feature_count[item]+hyper.alpha+hyper.beta))
        test_feature_score.append(score)
    train[feature + "_score"] = train_feature_score
    train[feature + "_score"][train[feature]=="-1"]=-1.0
    predict[feature + "_score"] = test_feature_score
    predict[feature + "_score"][predict[feature]=="-1"]=-1.0
    return train, predict

train,predict = nlp_feature_score("appIdAction", train, predict)
#train,predict = nlp_feature_score("interest1", train, predict)

print(train.shape, predict.shape)

data = pd.concat([train, predict])

# 分布统计特征
aid_age_count = data.groupby(['aid', 'age']).size().reset_index().rename(columns={0: 'aid_age_count'})
data = pd.merge(data, aid_age_count, 'left', on=['aid', 'age'])
aid_gender_count = data.groupby(['aid', 'gender']).size().reset_index().rename(columns={0: 'aid_gender_count'})
data = pd.merge(data, aid_gender_count, 'left', on=['aid', 'gender'])

# 活跃特征
add = pd.DataFrame(data.groupby(["campaignId"]).aid.nunique()).reset_index()
add.columns = ["campaignId", "campaignId_active_aid"]
data = data.merge(add, on=["campaignId"], how="left")

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus',
                 'advertiserId','campaignId','creativeId','adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2',
                'interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train=data[data.label!=-1]
train_y=train.pop('label')

test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)

enc = OneHotEncoder()
raw_feature = ['creativeSize',"aid_age_count",'aid_gender_count','campaignId_active_aid','appIdAction_score']
train_x=train[raw_feature]
test_x=test[raw_feature]

for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')


# online
def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', verbose=100, early_stopping_rounds=2000)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('submission.csv', index=False)
    return clf
model=LGB_predict(train_x,train_y,test_x,res)
