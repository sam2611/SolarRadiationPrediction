from matplotlib import pyplot
import pandas as pd
import numpy as np

df = pd.read_csv("SolarPrediction.csv")

print(df.head())
from sklearn.model_selection import train_test_split
X = df[['Temperature', 'Pressure','Humidity', 'WindDirection(Degrees)', 'Speed']] #Independent variable
y = df['Radiation'] #dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#model building
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)

predictions = lm.predict(X_test)
sse = np.sum((predictions - y_test)**2)
sst = np.sum((y_test - y_test.mean())**2)
R_square = 1 - (sse/sst)
print('R square obtain for normal equation method is :',R_square)

plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions));

f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test,predictions,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test - predictions),ax=ax,color='b')
ax.axvline((y_test - predictions).mean(),color='k',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
