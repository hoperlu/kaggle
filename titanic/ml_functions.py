import numpy as np
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

	def process_features(self,**kwargs):
		'''
		format of kwargs: label name of column=('method_1','method_2')
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
		function: apply function on this column
		tip: discard會讓資料變少，所以應該先把要用其他method的col處理完，最後再處理要discard的col
		'''
		discarded_tr=[]#record discarded rows, so we can discard corresponding rows of label later
		discarded_te=[]
		for key in kwargs:
			train,test=kwargs[key][0](self.train[key],self.test[key])
			train,test=kwargs[key][1](train,test)
			kwargs[key].extend(train,test)
		discarded_tr=set(discarded_tr)
		discarded_te=set(discarded_te)
		for key in kwargs:
			kwargs[key][2].drop(discarded_tr,0,inplace=True)
			kwargs[key][3].drop(discarded_te,0,inplace=True)
		train=pd.DataFrame(0,len(self.train)-len(discarded_tr),'temp_index')
		for key in kwargs:
			train=pd.concat([train,kwargs[key][2]])
		train.drop('temp_index',1,inplace=True)
		test=pd.DataFrame(0,len(self.test)-len(discarded_te),'temp_index')
		for key in kwargs:
			test=pd.concat([test,kwargs[key][3]])
		test.drop('temp_index',1,inplace=True)
		label





	if type(col)==type(''): #transform col into integer index
		col=list(self.values.columns).index(col)
	if method=='discard':
		deleted=[]
		for index in range(self.values.shape[0]):
			if pd.isna(self.values.iat[index,col]):
				deleted.append(index)
		return self.values.drop(deleted).reset_index().drop('index',1)
	res=self.values.copy()
	#if method=='DataWig':
	if method=='mean':
		mean=res.iloc[:,col].mean()
		for index in range(res.shape[0]):
			if pd.isna(res.iat[index,col]):
				res.iat[index,col]=mean
		return res
	for index in range(res.shape[0]):
		if pd.isna(res.iat[index,col]):
			res.iat[index,col]=method
	return res
