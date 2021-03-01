import ml_functions
import pandas as pd

#train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#ml_functions.missing_values_inspector(train)
#ml_functions.missing_values_inspector(test)

print(test.iloc[:4,1])