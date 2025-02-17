%pip install seaborn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
model=LinearRegression()
df=pd.read_csv("homeprices.csv")
df
median_bedrooms = df['bedrooms'].median()
print(median_bedrooms)
df['bedrooms']=df['bedrooms'].fillna(median_bedrooms)
df
df.describe()
corr=df.corr()
df.corr()
plt.figure(figsize=(10, 6))

# Plot the correlation matrix using imshow
plt.imshow(corr, cmap='RdBu', interpolation='nearest')
plt.colorbar()

# Set axis labels and title
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Matrix')

# Show the plot
plt.show()
model.fit(df[['area','bedrooms','age']],df.price)
model.predict([[2500,4,5]])
%pip install joblib
import joblib
joblib.dump(model,'model_joblib')
mj=joblib.load('model_joblib')
mj.predict([[2500,4,5]])