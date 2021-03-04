#from main import DATA,data_inspector
import pandas as pd
import numpy as np

train=pd.read_csv('train.csv')

'''
train=train.values
for index in range(train.shape[1]):
	print(type(train[0,index]))
'''
'''
df = pd.DataFrame(np.arange(30).reshape(6, 5),columns=['A', 'B', 'C', 'D','E'])
df=Data(df)
df.input_.iat[1,1]=pd.NA
df.input_.iat[2,3]=pd.NA
print(df.input_,'\n')

df=Data(df.process_missing_values(3,'mean'))
print(df.input_,'\n')
df=Data(df.process_missing_values(1,'discard'))

print(df.input_,'\n')

train=Data(pd.read_csv('train.csv'))
test=Data(pd.read_csv('test.csv'))
train.missing_values_inspector()
test.missing_values_inspector()
'''