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
			print('{:>10}: {:>10},{:>10.2f}%'.format(col,count,count/len(data)*100))
	print('')

class DATA():
	def __init__(self,train,label,test):
		self.label=train[label] #series
		self.train=train.drop(label,1)#dataframe
		self.test=test.copy()#dataframe

	def process_features(self,label_type,**kwargs):
		'''
		label_type='continuous' or 'categorical'
		format of kwargs: name of column=['method_1','method_2']
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
		mode: use mode of that column to replace missing values
		integer: use integer to replace missing values
		extra: add an extra category to represent missing values(only suitable for categorical features)
		function: apply function on this column
		do_nothing: do nothing. only suitable for algorithms which can deal with missing values
		tip: discard會讓資料變少，所以應該先把要用其他method的col處理完，最後再處理要discard的col
		困難點: 要用ML方法推論missing values，要先把非目標欄位都轉成numerical，所以非目標欄位的missing value怎處理
		'''
		discarded_tr=[]#record discarded rows, so we can discard corresponding rows of label later
		discarded_te=[]
		for key in self.train.columns:#input series, output dataframe
			train,test=kwargs[key][0](key,self.train[key],self.test[key])
			kwargs[key].extend([train,test])
		for key in self.train.columns:#input dataframe, output dataframe
			if kwargs[key][1].__name__=='discard':
				discard(kwargs[key][2],kwargs[key][3],discarded_tr,discarded_te)
			else:
				kwargs[key][2],kwargs[key][3]=kwargs[key][1](key,kwargs[key][2],kwargs[key][3])
			kwargs[key][2][key],kwargs[key][3][key]=normalize(key,kwargs[key][2],kwargs[key][3])
		discarded_tr=set(discarded_tr)
		discarded_te=set(discarded_te)
		train=pd.DataFrame(0,range(len(self.train)),['t_em_p_i_n_dex'],dtype=float)
		train=pd.concat([train].extend([kwargs[key][2] for key in self.train.columns]),1)
		train.drop('t_em_p_i_n_dex',1,inplace=True)
		train.drop(discarded_tr,0,inplace=True)
		test=pd.DataFrame(0,range(len(self.test)),['t_em_p_i_n_dex'],dtype=float)
		test=pd.concat([test].extend([kwargs[key][3] for key in self.train.columns]),1) #join=‘inner’ or ‘outer’
		test.drop('t_em_p_i_n_dex',1,inplace=True)
		test.drop(discarded_te,0,inplace=True)
		if label_type=='categorical':
			pass
		self.label.drop(discarded_tr,inplace=True)
		return train.values,test.values,label.values

def continuous(key,train,test):
	#input series, return DataFrame
	return pd.DataFrame(train,range(len(train)),[key],dtype=float),pd.DataFrame(test,range(len(test)),[key],dtype=float)

def categorical(key,train,test):
	#input series, every unique term is a column, return DataFrame
	name={}
	count=0
	for term in train:
		if not pd.isna(term) and term not in name:
			name[term]=count
			count+=1
	res_tr=pd.DataFrame(0,range(len(train)),[key for index in range(len(name))],dtype=float)
	for index in range(len(train)):
		if pd.isna(train[index]):
			res_tr.iloc[index,:]=np.nan
		else:
			res_tr.iat[index,name[train[index]]]=1
	res_te=pd.DataFrame(0,range(len(test)),[key for index in range(len(name))],dtype=float)
	for index in range(len(test)):
		if pd.isna(test[index]):
			res_te.iloc[index,:]=np.nan
		else:
			res_te.iat[index,name[test[index]]]=1
	return res_tr,res_te

def discard(train,test,discarded_tr,discarded_te):
	#discard missing values. 遺棄不是在此函式內發生，此函式僅記錄要遺棄那些rows
	for index in range(len(train)):
		if pd.isna(train.iat[index,0]):
			discarded_tr.append(index)
	for index in range(len(test)):
		if pd.isna(test.iat[index,0]):
			discarded_te.append(index)

def datawig(key,train,test):
	pass

def mean(key,train,test):
	#input dataframe, output dataframe. use mean to substitute missing values
	res_tr=train.copy()
	mean=res_tr[key].mean()
	for index in range(len(res_tr)):
		if pd.isna(res_tr.iat[index,0]):
			res_tr.iloc[index,:]=mean
	res_te=test.copy()
	for index in range(len(res_te)):
		if pd.isna(res_te.iat[index,0]):
			res_te.iloc[index,:]=mean
	return res_tr,res_te

def median(key,train,test):
	#input dataframe, output dataframe. use median to substitute missing values
	res_tr=train.copy()
	median=res_tr[key].median()
	for index in range(len(res_tr)):
		if pd.isna(res_tr.iat[index,0]):
			res_tr.iloc[index,:]=median
	res_te=test.copy()
	for index in range(len(res_te)):
		if pd.isna(res_te.iat[index,0]):
			res_te.iloc[index,:]=median
	return res_tr,res_te

def mode():
	pass

def extra(key,train,test):
	#only suitable for categorical
	res_tr=train.copy()
	for term in res_tr[0].isna():
		if term:
			temp=pd.DataFrame(0,range(len(res_tr)),[0],dtype=float)
			value=max(res_tr[0])
			for index in range(len(res_tr)):
				temp.iat[index,0]=value
			res_tr=pd.concat([res_tr,temp],1)
			break
	res_te=test.copy()
	for term in res_te[0].isna():
		if term:
			temp=pd.DataFrame(0,range(len(res_te)),[0],dtype=float)
			value=max(res_te[0])
			for index in range(len(res_te)):
				temp.iat[index,0]=value
			res_te=pd.concat([res_te,temp],1)
			break
	return res_tr,res_te

def do_nothing(key,train,test):
	pass

def integer():
	pass

def normalize(key,train,test):
	#input dataframe, output key values of dataframe
	mean=train[key].mean()
	std=train[key].std()
	return (train[key]-mean)/std,(test[key]-mean)/std

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
ml_functions.data_inspector(train)
ml_functions.data_inspector(test)

train.drop(['PassengerId'],1,inplace=True)
test.drop('PassengerId',1,inplace=True)

data=ml_functions.DATA(train,'Survived',test)
del train
del test
train,test,label=data.process_features(Age=[continuous,discard],Sex=[categorical,mean])
#now, train and test are feeding data