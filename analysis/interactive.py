import numpy as np
from scipy.stats import norm, expon, chisquare
import sys
import matplotlib.pyplot as plt
import tabulate
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import math
import pandas as pd




objective_data = np.genfromtxt("../data/clear_objectives.txt", delimiter=',')
effective_data = np.genfromtxt("../data/effective_communicator.txt", delimiter=',')


X = objective_data
y = effective_data


if len(X.shape) == 1:
    print("Reshaping")
    X = np.array([X]).T
plt.style.use('ggplot')
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 

plt.scatter(X, y)
plt.title('Clear Objectives vs Clear Instruction')
plt.xlabel('Course objectives clearly defined rating [1, 5]')
plt.ylabel('Instructor preparedness rating [1, 5]')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn


lm = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
lm.fit(X_train, y_train)

# PLOTS

plt.scatter(y_test, lm.predict(X_test))
plt.title('Clear Objectives vs Clear Instruction')
plt.xlabel('Instructor preparedness [1, 5]')
plt.ylabel('Predicted instructor preparedness [1, 5]')
plt.show()


plt.scatter(X_test, y_test)
plt.plot(X_test, lm.predict(X_test), color='y')
plt.title('Clear Objectives vs Clear Instruction')
plt.xlabel('Course objectives clearly defined rating [1, 5]')
plt.ylabel('Instructor preparedness rating [1, 5]')
plt.show()


plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, c='y', label='Train')
plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, label='Test')
plt.title('Clear Objectives vs Clear Instruction')
plt.legend(loc='upper right')
plt.show()

from sklearn.metrics import mean_squared_error
print("MSE: ", mean_squared_error(lm.predict(X_test), y_test))
print("Fit a model X_train, and calculate MSE with Y_train:", np.mean((y_train - lm.predict(X_train)) ** 2))
print("Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((y_test - lm.predict(X_test)) ** 2))

print("Estimated line: y = %0.4fx + %0.4f" %(lm.coef_, lm.intercept_))


from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error
from sklearn.metrics import explained_variance_score, mean_absolute_error

r2 = r2_score(y_test, lm.predict(X_test))

print(r2)
print(lm.score(X_test, y_test))


print(median_absolute_error(y_test, lm.predict(X_test)))
print(mean_squared_error(y_test, lm.predict(X_test)))
print(explained_variance_score(y_test, lm.predict(X_test)))
print(mean_absolute_error(y_test, lm.predict(X_test)))
