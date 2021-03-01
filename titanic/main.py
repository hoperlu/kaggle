import ml_functions
import pandas as pd
import numpy as np

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

label=train.iloc[:,1]
train.drop(['Survived','PassengerId'],1,inplace=True)
test.drop('PassengerId',1,inplace=True)

all_=train.append(test)
all_=ml_functions.process_missing_values()