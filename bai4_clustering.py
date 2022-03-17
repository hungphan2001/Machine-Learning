import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import spatial

filename = "iris.csv"

#1. Đọc file iris.csv ghi vào frame df_iris.
df_iris = pd.read_csv(filename)
##df_iris = df_iris.drop(["Id"],axis = 1)
print (df_iris)
print('-----------------------------------------------------------------------')

#2. Trộn ngẫu nhiên df_iris và in ra 10 dòng đầu tiên.
n = sum(1 for line in open(filename)) - 1
df_iris_mix=df_iris.sample(n=n)
print(df_iris_mix.head(10))
print('-----------------------------------------------------------------------')

#3. Gán X là 4 cột (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) của df_iris. Hiện thị 10 dòng đầu tiên của X.
X = df_iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
print(X.head(10))
print('-----------------------------------------------------------------------')

#4. Gán y là cột cuối cùng của df_iris. Hiện thị 10 dòng đầu tiên của y.
y = df_iris[['Species']]
# y = df_iris.iloc[: , -1]
print(y.head(10))
print('-----------------------------------------------------------------------')

#5. Hiện thị các lớp có trong y và số lượng tương ứng.
number_in_classes = df_iris['Species'].value_counts()
print(number_in_classes)
print('-----------------------------------------------------------------------')

#6. Tạo mô hình học model_kmeans: huấn luyện tập X với kỹ thuật học máy Kmeans (số cụm = 3).
n_cluster = 3
model_kmeans = KMeans(n_clusters=n_cluster)
model_kmeans.fit(X.values)
print('-----------------------------------------------------------------------')

#7. Hiện thị các tâm cụm tìm được.
plt.scatter(X.iloc[ : , 0], X.iloc[ :, 1],c=model_kmeans.labels_, s=50, cmap='viridis')
centers = model_kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black',s=200, alpha=0.5)
print(centers)
plt.show()
print('-----------------------------------------------------------------------')

#8. Hiện thị điểm dữ liệu X5 (dòng số 5 của X) và nhãn y5 tương ứng.
X_5 = X.iloc[4 , :]
print('X_5')
print(X_5)
y_5 = y.iloc[4 , :]
print('y_5')
print(y_5)
print('-----------------------------------------------------------------------')

#9. Tìm cụm gần điểm X5 nhất.
plt.scatter(X.iloc[ : , 0], X.iloc[ :, 1],c=model_kmeans.labels_, s=50, cmap='viridis')
plt.scatter(X.iloc[ 4 , 0], X.iloc[4 , 1],c='red', s=70, cmap='viridis',marker='*')
plt.scatter(centers[:, 0], centers[:, 1], c='black',s=200, alpha=0.5)

print(model_kmeans.labels_)
predicted_class = model_kmeans.predict([list(X_5)])
print(centers[predicted_class, 0:2])
plt.show()
print('-----------------------------------------------------------------------')
