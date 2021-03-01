import pandas as pd
import numpy as np

def missing_values_inspector(data):
	#inspect whether data have missing values, if so, print column names of these missing values
	for col in data.columns:
		count=0
		for term in pd.isna(data[col]):
			if term:
				count+=1
		if count!=0:
			print(col,count)

def process_missing_values(data,col,method):
	'''
	return processed column of data
	col can be label or index of column
	if method is discard, discard data with missing values
	if method is ML method, infer missing values with that ML method
	if method is mean, use mean of that column to replace missing values
	else, use method to replace missing values (here, method should be a number)
	tip: discard會讓資料變少，所以應該先把要用其他method的col處理完，最後再處理要discard的col
	'''
	if type(col)==type(''): #transform col into integer index
		col=list(data.columns).index(col)
	if method=='discard':
		deleted=[]
		for index in range(data.shape[0]):
			if pd.isna(data.iat[index,col]):
				deleted.append(index)
		return data.drop(deleted).reset_index().drop('index',1)
	res=data.copy()
	#if method=='ML':
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
