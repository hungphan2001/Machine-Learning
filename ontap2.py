# Dạng 3. Phân cụm với K-means
# 1a (2đ). Đọc vào file car data.csv được biến dataframe df. Hiển thị df.
# 1b (2đ). Sử dụng pháp chuyển đổi Label encoder, tạo cột mới Fuel_Type_number của df để chuyển cột Fuel_Type gía trị chữ thành giá trị số.
# Hiển thị df.
# 1c (2đ). Từ dataframe df, trích chọn dataframe X gồm các cột Selling_Price,Present_Price,Kms_Driven,Fuel_Type_number.
# Hiển thị X.
# 1d (2đ). Phân cụm tập dữ liệu X thành 5 cụm bằng thuật toán K-means. In tâm của các cụm thu được.
# 1e (2đ). In chỉ số cụm chứa phần tử thứ 5 của tập X tức là phần từ X[4,:].

import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
filename='car data.csv'

# a
df = pd.read_csv(filename)
# print(df)
print('-------------------------------------------------------------------')

# b
data_Fuel_Type = df['Fuel_Type'].unique()
# print(data_Fuel_Type)
df['Fuel_Type']= label_encoder.fit_transform(df['Fuel_Type'])
data_Fuel_Type = df['Fuel_Type'].unique()
df = df.rename(columns=({'Fuel_Type':'Fuel_Type_number'}))
# print(data_Fuel_Type)
# print(df)
print('-------------------------------------------------------------------')

# c
X = df[['Selling_Price','Present_Price','Kms_Driven','Fuel_Type_number']]
# print(X)
print('-------------------------------------------------------------------')

# d
n_cluster = 5
model_kmeans = KMeans(n_clusters=n_cluster)
model_kmeans.fit(X.values)
print(model_kmeans.labels_)
print('-------------------------------------------------------------------')

# e
X_5 = X.iloc[4 , :]
print('X_5')
print(X_5)
predicted_class = model_kmeans.predict([list(X_5)])
print(predicted_class)
print('-------------------------------------------------------------------')
