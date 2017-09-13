#To support both python 2 and 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

#chieu cao
X = np.array([[142, 149, 153, 158, 160, 163, 166, 170, 173, 176, 179, 180, 182]]).T
#can nang
Y = np.array([[48, 48, 50, 52, 54, 55, 58, 60, 62, 64, 66, 67, 68]]).T

#building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis =1)

#Tinh toan trong luong cua fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
w = np.dot(np.linalg.pinv(A), b)
print('w=', w)
#preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

#hien thi
plt.plot(X.T, Y.T, 'ro')  # data
plt.plot(x0,y0)  #the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Chieu cao (cm)')
plt.ylabel('can nang (kg)')
plt.show()

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print (u'Du doan can nang cua nguoi co chieu cao 155 cm: %.2f (kg)' %(y1))
print (u'Du doan can nang cua nguoi co chieu cao 160 cm: %.2f (kg)' %(y2))

#fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, Y)

#so sanh ket qua
print (u'Nghiem tim duoc tu phuong trinh scikit-learn : ' , regr.coef_ )
print (u'Nghiem tim duoc tu phuong trinh (5):  ' , w.T)


