import ml_functions
import pandas as pd
import numpy as np
'''
df = pd.DataFrame(np.arange(12).reshape(3, 4),columns=['A', 'B', 'C', 'D'])
df.iat[1,1]=pd.NA
df.iat[2,3]=pd.NA
print(df,'\n')
a=ml_functions.process_missing_values(df,3,'mean')
print(a,'\n')
a=ml_functions.process_missing_values(a,1,'discard')

print(a,'\n')
'''
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
ml_functions.missing_values_inspector(train)
ml_functions.missing_values_inspector(test)