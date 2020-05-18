import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

dataset = pd.read_csv('cereals.csv',sep=',', encoding='Windows-1251')
y = np.array(dataset['Пищевая ценность'].values)
x = np.array(dataset['Сахар г'].values).reshape((-1, 1))

plt.scatter(x, y, s = 10, color = 'blue', alpha = 0.75) #построение графика распределения данных

model = LinearRegression().fit(x, y) #обучение модели

yy = model.intercept_ + model.coef_ * x #поиск уравнения линейной регрессии

plt.plot(x, yy, linewidth=1, color='black') #построение графика линейной регрессии
plt.show()

err = (mean_squared_error(y, yy)*len(x))/(len(x)-2) #среднеквадратичная ошибка
y_sr = sum(y)/len(x) #поиск среднего значения y
Q_r = sum((yy-y_sr)**2) #регрессионная квадратичная сумма
Q_e = 0 #квадратичная сумма ошибки

for i in range(0, len(x)):
    Q_e += (y[i]-yy[i])**2

Q = Q_r+Q_e #общая квадратичная сумма
print('Среднее значение выходной переменной: ', y_sr)
print ('Qr: ', Q_r, ', Qe: ', Q_e, ', Q: ', Q)

S_b1 = sqrt(err)/sqrt(sum([xs * xs for xs in x]) - ((sum(x)) ** 2 / len(x))) #степень изменчивости b1

t = model.coef_/S_b1 #t-критерий
f = Q_r*(len(x)-2)/Q_e #f-критерий

print('Коэффициенты регрессии: b0 = ', model.intercept_, ' и b1 = ', model.coef_)
print('Коэффициент детерминации R^2: ', r2_score(y, yy), ' или ', Q_r/Q)
print('Коэффициент корреляции R: ', sqrt(r2_score(y, yy)))
print('Значение среднеквадратической ошибки: ', err, ' или ', Q_e/(len(x)-2))
print('Стандартная ошибка оценивания: ', sqrt(err))
print('t-критерий: ', t)
print('F-статистика', f)



