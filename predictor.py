import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Book1.csv')
print(df.head())
print(df.info())

df.hist(bins=50,figsize=(20,15))
plt.show()

# Train - Test Splitting
import numpy as np
def train_test_split(data,test_ratio):
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]

train_set , test_set = train_test_split(df,0.2)
print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}")