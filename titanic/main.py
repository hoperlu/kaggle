import pandas as pd
import ml_functions
#imbalance data problem

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
ml_functions.data_inspector(train)
ml_functions.data_inspector(test)

#'Cabin','Name','Ticket'怎處理
train.drop(['PassengerId','Cabin','Name','Ticket'],1,inplace=True)
test.drop(['PassengerId','Cabin','Name','Ticket'],1,inplace=True)

data=ml_functions.DATA(train,'Survived',test)
del train
del test 
train,test,label=data.process_features(categorical,Pclass={'method_1':'categorical','method_2':'do_nothing'},
	Sex={'method_1':'categorical','method_2':'do_nothing'},Age={'method_1':'continuous','method_2':'mean'},
	SibSp={'method_1':'continuous','method_2':'do_nothing'},Parch={'method_1':'continuous','method_2':'do_nothing'},
	Fare={'method_1':'continuous','method_2':'median'},Embarked={'method_1':'categorical','method_2':'extra'})
#now, train and test are feeding data
