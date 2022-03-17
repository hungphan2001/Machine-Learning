import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1
df_diabetes = pd.read_csv("diabetes.csv")

# result = df_diabetes.head(10)
# print(result)

# 2
X = df_diabetes.drop(["Outcome"],axis = 1)
y = df_diabetes['Outcome']
# print('x', X.shape)
# print('y',y.shape)

# 3
print(df_diabetes.iloc[5])
print('X')
print(X.iloc[5])
print('y')
print(y.iloc[5])

# 4
# max = df_diabetes['Age'].max()
# print('max:',max)
# min = df_diabetes['Age'].min()
# print('min:',min)
# mean = df_diabetes['Age'].mean()
# print('mean:',mean)
# var = df_diabetes['Age'].var()
# print('var:',var)
# std = df_diabetes['Age'].std()
# print('std:',std)

# 5
# df_min_max_scaled = X.copy()
# column = 'Insulin'
# df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
# #display(df_min_max_scaled)
# max = df_min_max_scaled['Insulin'].max()
# print(max)
# min =df_min_max_scaled['Insulin'].min()
# print(min)

# 6
# X2 = df_diabetes[['BMI','Age']]
# X0 = X2[y == 0]
# X1 = X2[y == 1]
# plt.plot(X0['BMI'], X0['Age'], 'b^', markersize=4, alpha=.8)
# plt.plot(X1['BMI'], X1['Age'], 'go', markersize=4, alpha=.8)
# plt.xlabel('BMI')
# plt.ylabel('Age')
# plt.title('Nhãn lớp 0 và lớp 1')
# plt.plot()
# plt.show()

# 7
# print ('Tổng số phần tử 0 là', y[y == 0].shape[0])
# print ('Tổng số phần tử 1 là', y[y == 1].shape[0])
# y.value_counts()

# 8
# pca_breast = PCA(n_components=2)
# #display(x)
# Xnew = pd.DataFrame(data = pca_breast.fit_transform(X))
# #display(xnew)
# X0new = Xnew[y == 0]
# X1new = Xnew[y == 1]
# plt.plot(X0new[0], X0new[1], 'b^', markersize=4, alpha=.8)
# plt.plot(X1new[0], X1new[1], 'go', markersize=4, alpha=.8)
# plt.xlabel('pcax0')
# plt.ylabel('pcax1')
# plt.title('Nhãn lớp 0 và lớp 1')
# plt.plot()
# plt.show()

# 9
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#log_model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
#y_pred = log_model.predict(x_test)
#print(y_pred)
#print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)