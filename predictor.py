# Importing Libraries 
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv('Book1.csv')
print(df.head())
print(df.info())

df.hist(bins=50,figsize=(20,15))
plt.show()

# Train - Test Splitting
import numpy as np
def train_test_split(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]

train_set , test_set = train_test_split(df,0.2)
print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}")

''' Stratified Sampling - mtlb agr hmne jaise data ko training and testing mein divide kra toh ab yeh ho skta hai ki usko ek hi type ki 
values mil jaye mtlb agar ek field ki values 0 and 1 hai toh kya pta training wale ke pass saari 0 chali jaye toh usse bachne ke liye
use krte hai stratified sampling.
'''
# Stratified Sampling
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42) # n_splits means one set for training and the other for testing
for train_index , test_index in split.split(df , df['CHAS']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

print(strat_test_set.info)
