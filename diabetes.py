import numpy as np
import pandas as pd
data_train=np.genfromtxt("C:/Users/prashansa sinha/OneDrive/Desktop/ML_notes/_training_diabetes_x_y_train.csv",delimiter=",")
data_test=np.genfromtxt("C:/Users/prashansa sinha/OneDrive/Desktop/ML_notes/_test_diabetes_x_test.csv",delimiter=",")
x_train=data_train[:,:10]

y_train=data_train[:,10].reshape(-1,1)
print(x_train.shape)
print(y_train.shape)
x_test=data_test
x_test.shape
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
 #y_1pred=f"{y_pred:.5f}"
score_test=reg.score(x_test,y_pred)
score_test
print(y_pred)
np.savetxt('pred.csv',y_pred,delimiter=',',fmt='%.5f')