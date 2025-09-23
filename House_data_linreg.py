import numpy as np
import pandas as pd


#from subprocess import check_output
#print(check_output(['ls', '../input']).decode('utf8'))


import matplotlib.pyplot as plt

#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=np.inf)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\aashutosh\OneDrive\Desktop\mlprac\House_data.csv')
space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1,1)
y = np.array(price).reshape(-1,1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

pred = regressor.predict(xtest)

plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('Visuals for Training set')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('visuals for test set')
plt.xlabel('space')
plt.ylabel('price')
plt.show()

plt.scatter(xtest,ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('visuals for test set')
plt.xlabel('space')
plt.ylabel('price')
plt.show()