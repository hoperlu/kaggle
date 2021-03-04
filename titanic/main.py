import numpy as np
import pandas as pd
#imbalance data problem

def data_inspector(data):
	'''
	print data.shape and data.columns
	inspect whether data has missing values
	if so, print index of column, number and percentage of missing values
	'''
	print(data.shape)
	print(data.columns)
	for col in data.columns:
		count=0
		for term in pd.isna(data[col]):
			if term:
				count+=1
		if count!=0:
			print('{:>10}: {:>10},{:>10.2f}%'.format(col,count,count/l*100))
	print('')

class DATA():
	def __init__(self,train,label,test):
		self.label=train[label]
		self.train=train.drop(label,1)
		if self.train.columns==test.columns:
			self.test=test.copy()
		else:
			self.test=pd.DataFrame(0,index=range(len(test)),columns=self.train.columns)
			for col in self.train.columns:
				self.test[col]=test[col]

	def process_features(self,label_type,**kwargs):
		'''
		label_type='continuous' or 'categorical'
		format of kwargs: name of column=('method_1','method_2')
		kwargs每個key的value為list，在input時只有兩項，處理完變成
		('method_1','method_2',processed features of train,processed features of test)
		return feeding data
		method_1 is the attribute of the column
		continuous: the column is continuous feature
		categorical: the column is categorical feature
		function: apply function on this column
		method_2 is how we deal with missing values of the column
		discard: discard values with missing values
		ML methods: infer missing values with that ML method
		mean: use mean of that column to replace missing values
		median: use median of that column to replace missing values
		integer: use integer to replace missing values
		extra: add an extra category to represent missing values(only suitable for categorical features)
		function: apply function on this column
		tip: discard會讓資料變少，所以應該先把要用其他method的col處理完，最後再處理要discard的col
		'''
		discarded_tr=[]#record discarded rows, so we can discard corresponding rows of label later
		discarded_te=[]
		for key in kwargs:
			train,test=kwargs[key][0](self.train[key],self.test[key])
			if kwargs[key][1].__name__!='discard':
				train,test=kwargs[key][1](train,test)#input may be series or dataframe
			else:
				train,test=discard(train,test,discarded_tr,discarded_te)#input may be series or dataframe
			kwargs[key].extend([train,test])
		discarded_tr=set(discarded_tr)
		discarded_te=set(discarded_te)
		train=pd.DataFrame(0,range(len(self.train)),['temp_index'])
		train=pd.concat([train].extend([kwargs[key][2] for key in kwargs]),1)
		train.drop('temp_index',1,inplace=True)
		train.drop(discarded_tr,0,inplace=True)
		test=pd.DataFrame(0,range(len(self.test)),['temp_index'])
		test=pd.concat([test].extend([kwargs[key][3] for key in kwargs]),1) #join=‘inner’ or ‘outer’
		test.drop('temp_index',1,inplace=True)
		test.drop(discarded_te,0,inplace=True)
		if label_type=='categorical':

		label.drop(discarded_tr,0,inplace=True)
		return train.values,test.values,label.values

def continuous(train,test):
	

def categorical(train,test)
	pass

def discard(train,test,discarded_tr,discarded_te):
	pass

def datawig(train,test):
	pass

def mean(train,test):
	pass

def median(train,test):
	pass

def extra(train,test):
	pass

def integer(train,test):
	pass

train=read_csv('train.csv')
test=read_csv('test.csv')
data_inspector(train)
data_inspector(test)

train.drop(['PassengerId'],1,inplace=True)
test.drop('PassengerId',1,inplace=True)

data=DATA(train,'Survived',test)
del train
del test
train,test,label=data.process_features(Age=[continuous,discard],Sex=[categorical,mean])
#now, train and test are feeding data