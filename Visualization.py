import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("SolarPrediction.csv")
# take a look at the dataset
df.head()

# summarize the data
df.describe()

cdf = df[['Temperature','Radiation','Pressure']]
cdf.head(9)

viz = cdf[['Temperature','Radiation','Pressure']]
viz.hist()
plt.show()

plt.scatter(cdf.Temperature, cdf.Radiation,  color='orange')
plt.xlabel("Temperature")
plt.ylabel("Radiation")
plt.show()

plt.scatter(cdf.Pressure, cdf.Radiation,  color='red')
plt.xlabel("Pressure")
plt.ylabel("Radiation")
plt.show()

