import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

data = pd.read_csv('real_estate_price_size.csv')

Y = data['price']
X1 = data['size']

plt.scatter(X1,Y)
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()

X = sm.add_constant(X1)
results = sm.OLS(Y,X).fit()
results.summary()

plt.scatter(X1,Y)
yhat = 223.1787*X1+101900
fig = plt.plot(X1,yhat,lw=4,c='orange',label='regresion line')
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()
