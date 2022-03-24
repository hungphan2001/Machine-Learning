import pandas as pd
from sklearn.cluster import KMeans
#1 Đọc  fashion.csv  vào fashion_df
fashion_df = pd.read_csv('fashion.csv')
print(fashion_df)

#2 Hiển thị tênvà số lượng các cột của fashion_df
print(fashion_df.columns)
print("Tổng số cột là ", len(fashion_df.columns))
print("Kích thước của fashion data là ", fashion_df.shape)

#3 Từfashion_df, gán dataframe X là tất cả các cột trừ cột label. Hiển thị X ở dòng 150.
X = fashion_df.drop(["label"], axis = 1)
X150 = X.iloc[150,:]
print(X150)

# Phân cụm tập dữ liệuX thành 10 cụm bằng thuật toán K-means.
print('Kmeans')
model = KMeans(n_clusters=10, random_state=0).fit(X)
centers = model.cluster_centers_

# 4. In tâm của các cụm
print("In tâm của các cụm thu được:\n")
print (centers)

# 5 Tính khoảngcách từ điểm X150 (là X ở dòng số 150) tới các tâm cụm. X150 gần tâm cụm nào nhất?
# 6.  Vẽ tập dữ liệu X và tâm của 10 cụm 