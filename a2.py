import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


'''
train_df=pd.read_csv('/home/twilighce/data/train.csv')
position_df=pd.read_csv('/home/twilighce/pre/position.csv')
train_pos=pd.concat([train_df.set_index('positionID'),position_df.set_index('positionID')],axis=1,join='inner')
train_pos.to_csv('/home/twilighce/data/train2.csv',index=False)

test_df=pd.read_csv('/home/twilighce/data/test.csv')
test_pos=pd.concat([test_df.set_index('positionID'),position_df.set_index('positionID')],axis=1,join='inner')
test_pos.to_csv('/home/twilighce/data/test2.csv',index=False)

train_df=pd.read_csv('/home/twilighce/data/train.csv')
ad_df=pd.read_csv('/home/twilighce/pre/ad.csv')
train_ad=pd.concat([train_df.set_index('creativeID'),ad_df.set_index('creativeID')],axis=1,join='inner')
train_ad.to_csv('/home/twilighce/data/train2.csv',index=False)

test_df=pd.read_csv('/home/twilighce/data/test.csv')
test_ad=pd.concat([test_df.set_index('creativeID'),ad_df.set_index('creativeID')],axis=1,join='inner')
test_ad.to_csv('/home/twilighce/data/test2.csv',index=False)
'''
train_df=pd.read_csv('/home/twilighce/data/train2.csv')
test_df=pd.read_csv('/home/twilighce/data/test2.csv')
'''
print(train_df.info())
print(test_df.info())
print(train_df.columns.values)

print(train_df.head())
print(test_df.columns.values)

g=sns.FacetGrid(train_df,col='label')
g.map(plt.hist,'positionType',alpha=.5,bins=20)
plt.show()
'''
X_train=train_df.drop(['label','adID','camgaignID','advertiserID','appID'],axis=1)
Y_train=train_df['label']
X_test=test_df.drop(['instanceID','label','adID','camgaignID','advertiserID','appID'],axis=1)


svc=SVC(kernel='rbf',probability=True)
svc.fit(X_train,Y_train)
Y_pred=svc.predict_proba(X_test)
'''
random_forest=RandomForestClassifier(n_estimators=1000000)
random_forest.fit(X_train,Y_train)
Y_pred=random_forest.predict_proba(X_train)

gbm=xgb.XGBClassifier(learning_rate=0.95,n_estimators=16000,max_depth=4,min_child_weight=2,gamma=1,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',scale_pos_weight=1).fit(X_train,Y_train)
Y_pred=xgb.predict_proba(X_test)
'''
submission=pd.DataFrame({'instanceID':test_df['instanceID'],'prob':Y_pred[:,0]})
submission.to_csv('/home/twilighce/data/submission.csv',index=False)

