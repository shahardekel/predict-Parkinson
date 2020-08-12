import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv



#A
data = pd.read_csv('parkinsons_updrs_data.csv', index_col=0)

#B
male=[]
male[:]=(n for n in data['sex'] if n==0)

female=[]
female[:]=(n for n in data['sex'] if n==1)

objects_B=('male','female')
y_pos_B=np.arange(len(objects_B))
performance_B=[len(male), len(female)]
plt.bar(y_pos_B, performance_B, width=0.4, align='center', color='orange')
for i in range(len(performance_B)):
    plt.text(x =y_pos_B[i] - 0.05, y =performance_B[i] + 1, s = performance_B[i], size = 10)
plt.xticks(y_pos_B, objects_B)
plt.title('How many males and females are in the data?')
plt.show()

#C

plt.hist(data['age'],rwidth=0.8, color='purple')
plt.title('The distribution of the ages')
plt.show()

#D
x=data.groupby('sex')['motor_UPDRS'].mean()
male_D=x[0]
female_D=x[1]
objects_D=('male','female')
y_pos_D=np.arange(len(objects_D))
performance_D=[male_D,female_D]
plt.bar(y_pos_D,performance_D,width=0.4,align='center', color='green')
for i in range(len(performance_D)):
    plt.text(x =y_pos_D[i] - 0.05, y =performance_D[i], s = performance_D[i], size = 12)
plt.xticks(y_pos_D, objects_D, size=12)
plt.title('the mean of the variable "motor_UPDRS" between different sexes')
plt.show()

#E
data.dropna()
matrix=data[['motor_UPDRS','age','sex','test_time','NHR','HNR','DFA']]
pd.plotting.scatter_matrix(matrix,figsize=(12,10), diagonal='kde')
plt.show()

#F
y=data.motor_UPDRS
x=data[['age','sex','test_time','NHR','HNR','DFA']]
lr_model = LinearRegression()
lr_model.fit(x, y)
LS_estimators=np.hstack([lr_model.intercept_,lr_model.coef_])
print('the LS estimators with sklearn built in function:')
print(LS_estimators)



#G
"""the function from Q2"""
def least_squares_estimator(explanatoryX, responsesY):
    rowsX = len(explanatoryX)
    x1 = np.ones((rowsX, 1))
    Xnew = np.hstack((x1, explanatoryX))
    XnewT = np.transpose(Xnew)
    xTx = np.matmul(XnewT, Xnew)

    xTx_inverse = inv(xTx)
    xTy = np.matmul(XnewT, responsesY)

    result = np.matmul(xTx_inverse, xTy)

    return result

result_G=least_squares_estimator(x,y)
print('the LS estimators with the function from Q2:')
print(result_G)