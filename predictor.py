import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Book1.csv')
print(df.head())
print(df.info())

df.hist(bins=50,figsize=(20,15))
plt.show()
