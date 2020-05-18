import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('cereals.csv', sep=',', encoding='Windows-1251')
y = np.array(dataset['Пищевая ценность'].values[0:35])
x = np.array(dataset[['Калий мг', 'Вес', '№ витрины', 'Полка1', 'Полка2']].values[0:35])
d = []
for i in range(0, 35):
    if x[i, 2] == 'N1':
        x[i, 3] = 1
        x[i, 4] = 0
    elif x[i, 2] == 'N2':
        x[i, 3] = 0
        x[i, 4] = 1
    else:
        x[i, 3] = 0
        x[i, 4] = 0
    x[i, 2] = len(x[i, 2])

print(x)
model = LinearRegression().fit(x, y)
yy = []
for i in range(0, 35):
    yy.append(model.intercept_)
    for j in range(0, 3):
        yy[i] += x[i, j] * model.coef_[j]
print('Коэффициенты регрессии: b0 = ', model.intercept_, ' и b1 = ', model.coef_)
print('Коэффициент детерминации R^2: ', r2_score(y, yy))
pass
