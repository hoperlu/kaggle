import pandas as pd

def missing_values_inspector(data):
	#inspect whether data have missing values, if so, print column names of these missing values
	for col in data.columns:
		for term in pd.isna(data[col]):
			if term:
				print(col)
				break

def process_missing_values(data,col,method):
	'''
	return processed column of data
	col can be name or index of column
	if method is ignore, discard data with missing values
	if method is ML method, infer missing values with that ML method
	'''
	