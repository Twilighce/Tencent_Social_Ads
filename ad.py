import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

'''
train_df=pd.read_csv('/home/twilighce/pre/train.csv')
user_df=pd.read_csv('/home/twilighce/pre/user.csv')
train_user=pd.concat([train_df.set_index('userID'),user_df.set_index('userID')],axis=1,join='inner').reset_index()
train_user.to_csv('/home/twilighce/data/train.csv',index=False)

test_df=pd.read_csv('/home/twilighce/pre/test.csv')
test_user=pd.concat([test_df.set_index('userID'),user_df.set_index('userID')],axis=1,join='inner')
test_user.to_csv('/home/twilighce/data/test.csv')
'''
#user_installedapps_df=pd.read_csv('/home/twilighce/pre/user_installedapps.csv')
#user_app_actions_df=pd.read_csv('/home/twilighce/pre/user_app_actions.csv')
#app_categories.csv_df=pd.read_csv('/home/twilighce/pre/app_categories.csv')
#ad_df=pd.read_csv('home/twilighce/pre/ad.csv')
#position_df=pd.read_csv('/home/twilighce/pre/position.csv')
#combine=[train_df,test_df]
'''
print(res.info(6))
print(res.head())
print(res.tail())
print(train_df.head(6))
print(train_df.tail())
'''

train_df=pd.read_csv('/home/twilighce/data/train.csv')
test_df=pd.read_csv('/home/twilighce/data/test.csv')
combine=[train_df,test_df]





'''
print(train_df.columns.values)
print(test_df.columns.values)
'''

#print(train_df.columns.values)
#print(train_df.head())
#print(train_df.info())
#print(test_df.info())
#print(train_df.describe(include=['O']))
#print(train_df[['residence','label']].groupby(['residence'],as_index=False).mean().sort_values(by='label',ascending=False))
#print(train_df[['connectionType','label']].groupby(['connectionType'],as_index=False).mean().sort_values(by='label',ascending=False))
#print(train_df[['telecomsOperator','label']].groupby(['telecomsOperator'],as_index=False).mean().sort_values(by='label',ascending=False))
'''
g=sns.FacetGrid(train_df,col='label')
g.map(plt.hist,'age',alpha=.5,bins=20)
plt.show()

g=sns.FacetGrid(train_df,col='label')
g.map(plt.hist,'education')
plt.show()
'''
#print('Before',train_df.shape,test_df.shape)
train_df=train_df.drop(['userID','conversionTime'],axis=1)
test_df=test_df.drop(['userID'],axis=1)
combine=[train_df,test_df]
#print('after',train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)

#print(train_df[['residence','label']].groupby(['residence'],as_index=False).mean())

#train_df['Band']=pd.cut(train_df['hometown'],34)
#print(train_df[['Band','label']].groupby(['Band'],as_index=False).mean().sort_values(by='Band',ascending=True))

train_df['hometown']=train_df['hometown']//100
test_df['hometown']=train_df['hometown']//100
train_df['residence']=train_df['residence']//100
test_df['residence']=test_df['residence']//100
#print(train_df.head(10))

#train_df['homeband']=pd.cut(train_df['hometown'],34,right=False,include_lowest=True)
#print(train_df[['homeband','label']].groupby(['homeband'],as_index=False).mean().sort_values(by='homeband',ascending=True))

m={0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:2,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:1,33:2,34:2}
combine=[train_df,test_df]
for dataset in combine:
	dataset['hometown']=dataset['hometown'].map(m)
	dataset['residence']=dataset['residence'].map(m).astype(int)
	dataset.loc[dataset['age']<=10,'age']=0
	dataset.loc[(dataset['age']>10)&(dataset['age']<=40),'age']=1
	dataset.loc[dataset['age']>40,'age']=2
	dataset['age']=dataset.astype(int)
'''
g=sns.FacetGrid(train_df,col='label')
g.map(plt.hist,'clickTime')
plt.show()
'''
#print(train_df.head())
'''
train_df['timeband']=pd.cut(train_df['clickTime'],5)
print(train_df[['timeband','label']].groupby(['timeband'],as_index=False).mean().sort_values(by='timeband',ascending=True))
'''
train_df=train_df.drop(['clickTime','gender','marriageStatus','haveBaby','telecomsOperator','education','residence'],axis=1)
#Y_train=train_df['label']
test_df=test_df.drop(['clickTime','gender','marriageStatus','haveBaby','telecomsOperator','education','residence'],axis=1)

train_df.to_csv('/home/twilighce/data/train2.csv',index=False)
test_df.to_csv('/home/twilighce/data/test2.csv',index=False)
#print(train_df.head())

#print(train_df.shape,test_df.shape)
'''
svc=SVC(kernel='linear',probability=True)
svc.fit(X_train,Y_train)
Y_pred=svc.predict_proba(X_test)
#print(Y_pred)
acc_svc=round(svc.score(X_train,Y_train)*100,2)
print(acc_svc)
submission=pd.DataFrame({'instanceID':test_df['instanceID'],'prob':Y_pred[:,0]})
submission.to_csv('/home/twilighce/data/submission2.csv',index=False)
'''
