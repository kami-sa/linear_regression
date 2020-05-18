import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('cereals.csv',sep=',', encoding='Windows-1251')
y = np.array(dataset['Пищевая ценность'].values[0:35])
x = np.array(dataset[['Калий мг','Вес']].values[0:35])
_x, _y, _z = [], [], []
for i in range(0, 35):
    _x.append(x[i, 0])
    _y.append(x[i, 1])
    _z.append(y[i])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(_x, _y, _z, c='r', marker='o')

ax.set_xlabel('Калий')
ax.set_ylabel('Вес')
ax.set_zlabel('Пищевая ценность')

model = LinearRegression().fit(x, y)
yy = []
for i in range(0, 35):
    for j in range(0, 1):
        yy.append(model.intercept_ + model.coef_[j] * x[i, j])
print('Коэффициенты регрессии: b0 = ', model.intercept_, ' и b1 = ', model.coef_)
ax.plot_trisurf(_x, _y, yy)
print('Коэффициент детерминации R^2: ', r2_score(y, yy))
print(mean_squared_error(y, yy))
plt.show()