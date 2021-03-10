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
		label_type=continuous or categorical
		format of kwargs: name of column={'method_1':method_1,'method_2':method_2}
		kwargs每個key的value為dictionary，在input時只有兩項，處理完至少有四項
		{'method_1','method_2','processed features of train','processed features of test'}
		return feeding data
		method_1是我們如何將column數值化, 預設有continuous和categorical，也能加入其他新function
		method_2 is how we deal with missing values of the column, 預設有discard,ML methods(infer missing values 
		with the ML method), mean, median, mode, extra, do_nothing, 也能加入其他新function
		tip: discard會讓資料變少，所以應該先把要用其他method的col處理完，最後再處理要discard的col
		困難點: 要用ML方法推論missing values，要先把非目標欄位都轉成numerical，所以非目標欄位的missing value怎處理
		'''
		discarded_tr=[]#record discarded rows, so we can discard corresponding rows of label later
		discarded_te=[]
		for key in self.train.columns:#input series, output dataframe
			train,test=kwargs[key]['method_1'](key,self.train[key],self.test[key])
			kwargs[key]['train']=train
			kwargs[key]['test']=test
		for key in self.train.columns:#input dataframe, output dataframe
			if kwargs[key]['method_2'].__name__=='discard':
				discard(kwargs[key]['train'],kwargs[key]['test'],discarded_tr,discarded_te)
			else:
				kwargs[key]['train'],kwargs[key]['test']=kwargs[key]['method_2'](key,kwargs[key]['train'],kwargs[key]['test'])
			kwargs[key]['train'][key],kwargs[key]['test'][key]=normalize(key,kwargs[key]['train'],kwargs[key]['test'])
		discarded_tr=set(discarded_tr)
		discarded_te=set(discarded_te)
		train=pd.concat([kwargs[key]['train'] for key in kwargs],1)
		train.drop(discarded_tr,0,inplace=True)
		test=pd.concat([kwargs[key]['test'] for key in kwargs],1)
		test.drop(discarded_te,0,inplace=True)
		self.label=label_type('label',self.label)
		self.label['label']=normalize('label',self.label)
		self.label.drop(discarded_tr,0,inplace=True)
		#print(train.shape,test.shape,self.label.shape)
		return train.values,test.values,self.label.values

def continuous(key,train,test='-1'):
	#input series, return DataFrame
	if type(test)!=str:
		return pd.DataFrame(train,range(len(train)),[key],dtype=float),pd.DataFrame(test,range(len(test)),[key],dtype=float)
	return pd.DataFrame(train,range(len(train)),[key],dtype=float)

def categorical(key,train,test='-1'):
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
	if type(test)!=str:
		res_te=pd.DataFrame(0,range(len(test)),[key for index in range(len(name))],dtype=float)
		for index in range(len(test)):
			if pd.isna(test[index]):
				res_te.iloc[index,:]=np.nan
			else:
				res_te.iat[index,name[test[index]]]=1
		return res_tr,res_te
	return res_tr

def discard(train,test,discarded_tr,discarded_te):
	#discard rows with missing values. 遺棄不是在此函式內發生，此函式僅記錄要遺棄那些rows
	for index in range(len(train)):
		if pd.isna(train.iat[index,0]):
			discarded_tr.append(index)
	for index in range(len(test)):
		if pd.isna(test.iat[index,0]):
			discarded_te.append(index)

def datawig(key,train,test):
	pass

def mean(key,train,test='-1'):
	#input dataframe, output dataframe. use mean to substitute missing values
	res_tr=train.copy()
	mean=res_tr[key].mean()
	print(pd.isna(res_tr))
	for index in range(len(res_tr)):
		print(res_tr.iloc[index,1])
	for index in range(len(res_tr)):
		if pd.isna(res_tr.iat[index,0]):
			res_tr.iloc[index,:]=mean
	if type(test)!=str:
		res_te=test.copy()
		for index in range(len(res_te)):
			if pd.isna(res_te.iat[index,0]):
				res_te.iloc[index,:]=mean
		return res_tr,res_te
	return res_tr

def median(key,train,test='-1'):
	#input dataframe, output dataframe. use median to substitute missing values
	res_tr=train.copy()
	median=res_tr[key].median()
	for index in range(len(res_tr)):
		if pd.isna(res_tr.iat[index,0]):
			res_tr.iloc[index,:]=median
	if type(test)!=str:
		res_te=test.copy()
		for index in range(len(res_te)):
			if pd.isna(res_te.iat[index,0]):
				res_te.iloc[index,:]=median
		return res_tr,res_te
	return res_tr

def mode(key,train,test='-1'):
	#input dataframe, output dataframe. use mode to substitute missing values
	res_tr=train.copy()
	if type(test)!=str:
		res_te=test.copy()
		if res_tr.shape[1]==1:#continuous
			mode=res_tr[key].mode()
			if len(mode)==1:
				for index in range(len(res_tr)):
					if pd.isna(res_tr.iat[index,0]):
						res_tr.iat[index,0]=mode
				for index in range(len(res_te)):
					if pd.isna(res_te.iat[index,0]):
						res_te.iat[index,0]=mode
			else:#when there are more than one mode, use the mode which is closet to mean
				mean=res_tr[key].mean()
				m=abs(mean-mode[0])
				i=0
				for index in range(1,len(mode)):
					if abs(mode[index]-mean)<m:
						i=index
						m=abs(mode[index]-mean)
				for index in range(len(res_tr)):
					if pd.isna(res_tr.iat[index,0]):
						res_tr.iat[index,0]=mode[i]
				for index in range(len(res_te)):
					if pd.isna(res_te.iat[index,0]):
						res_te.iat[index,0]=mode[i]
		else:#categorical
			s=res_tr[key].sum(0)
			m=max(s)
			for index in range(len(s)):
				if s.iat[index]==m:
					i=index
					break
			for index in range(len(res_tr)):
				if pd.isna(res_tr.iat[index,0]):
					res_tr.iloc[index,:]=0
					res_tr.iat[index,i]=1
			for index in range(len(res_te)):
				if pd.isna(res_te.iat[index,0]):
					res_te.iloc[index,:]=0
					res_te.iat[index,i]=1
		return res_tr,res_te
	if res_tr.shape[1]==1:#continuous
		mode=res_tr[key].mode()
		if len(mode)==1:
			for index in range(len(res_tr)):
				if pd.isna(res_tr.iat[index,0]):
					res_tr.iat[index,0]=mode
		else:#when there are more than one mode, use the mode which is closet to mean
			mean=res_tr[key].mean()
			m=abs(mean-mode[0])
			i=0
			for index in range(1,len(mode)):
				if abs(mode[index]-mean)<m:
					i=index
					m=abs(mode[index]-mean)
			for index in range(len(res_tr)):
				if pd.isna(res_tr.iat[index,0]):
					res_tr.iat[index,0]=mode[i]
	else:#categorical
		s=res_tr[key].sum(0)
		m=max(s)
		for index in range(len(s)):
			if s.iat[index]==m:
				i=index
				break
		for index in range(len(res_tr)):
			if pd.isna(res_tr.iat[index,0]):
				res_tr.iloc[index,:]=0
				res_tr.iat[index,i]=1
	return res_tr

def extra(key,train,test='-1'):
	#input dataframe, output dataframe, add an extra category to represent missing values(only suitable for categorical features)
	res_tr=train.copy()
	for index in range(len(res_tr)):
		if pd.isna(res_tr.iat[index,0]):
			temp=pd.DataFrame(0,range(len(res_tr)),[key],dtype=float)
			for index2 in range(len(res_tr)):
				if pd.isna(res_tr.iat[index2,0]):
					temp.iat[index2,0]=1
					res_tr.iloc[index2,:]=0
			res_tr=pd.concat([res_tr,temp],1)
			break
	if type(test)!=str:
		res_te=test.copy()
		for index in range(len(res_te)):
			if pd.isna(res_te.iat[index,0]):
				temp=pd.DataFrame(0,range(len(res_te)),[0],dtype=float)
				for index2 in range(len(res_te)):
					if pd.isna(res_te.iat[index2,0]):
						temp.iat[index2,0]=1
						res_te.iloc[index2,:]=0
				res_te=pd.concat([res_te,temp],1)
				break
		return res_tr,res_te
	return res_tr

def do_nothing(key,train,test='-1'):
	#input dataframe, output dataframe. really do nothing at all
	res_tr=train.copy()
	if type(test)!=str:
		res_te=test.copy()
		return res_tr,res_te
	return res_tr

def normalize(key,train,test='-1'):
	#input dataframe, output key values of dataframe
	mean=train[key].mean()
	std=train[key].std()
	if type(test)!=str:
		return (train[key]-mean)/std,(test[key]-mean)/std
	return (train[key]-mean)/std

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
data_inspector(train)
data_inspector(test)

#'Cabin','Name','Ticket'怎處理
train.drop(['PassengerId','Cabin','Name','Ticket'],1,inplace=True)
test.drop(['PassengerId','Cabin','Name','Ticket'],1,inplace=True)

data=DATA(train,'Survived',test)
del train
del test 
train,test,label=data.process_features(categorical,Pclass={'method_1':categorical,'method_2':do_nothing},
	Sex={'method_1':categorical,'method_2':do_nothing},Age={'method_1':continuous,'method_2':mode},
	SibSp={'method_1':continuous,'method_2':do_nothing},Parch={'method_1':continuous,'method_2':do_nothing},
	Fare={'method_1':continuous,'method_2':median},Embarked={'method_1':categorical,'method_2': mean})
#now, train and test are feeding data
