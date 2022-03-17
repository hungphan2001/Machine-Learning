# Dạng 1. Phân lớp nhị phân.
# 1a (2đ). Đọc vào file user_data.csv được biến dataframe df. Hiển thị df.
# 1b (2đ). Sử dụng pháp chuyển đổi Label encoder, tạo cột mới Gender_number của df để chuyển cột Gender gía trị chữ thành giá trị số.
# Hiển thị df.
# 1c(2đ). Từ dataframe df, trích chọn dataframe X gồm các cột Gender_numer,Age,EstimatedSalary
# ,giá trị nhãn y là cột Purchased, hiển thị X,y.
# 1d (1đ). Chia ngẫu nhiên X,y thành X_train,y_train và X_test,y_test theo tỉ lệ 80:20. Hiển thị X_test,y_test.
# 1e (2đ). Sử dụng một một trong các kỹ thuật phân lớp nhị phân sau:
# Logistics Regression, SVM, Adaboost để huấn luyện mô hình học máy với tập dữ liệu X_train,y_train.
# 1g (1đ). Dự báo kết quả và đánh giá độ chính xác dự báo trên tập X_test,y_test.

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# a
file_name = 'User_Data.csv'
df = pd.read_csv(file_name)
# print(df)
print('-------------------------------------------------------------------')

# b
data_gender = df['Gender'].unique()
# print(data_gender)
df['Gender']= label_encoder.fit_transform(df['Gender'])
data_gender = df['Gender'].unique()
df = df.rename(columns=({'Gender':'Gender_numer'}))
# print(data_gender)
# print(df)
print('-------------------------------------------------------------------')

# c
X = df[['Gender_numer','Age','EstimatedSalary']]
# print('X')
# print(X)
y = df[['Purchased']]
# print('y')
# print(y)
print('-------------------------------------------------------------------')

# d
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print('X_test')
# print(X_test.shape)
# print(X_test)
# print('y_test')
# print(y_test.shape)
print(y_test)
print('-------------------------------------------------------------------')

# e
log_model = LogisticRegression(max_iter=1000).fit(X_train ,y_train.values.ravel())
print('-------------------------------------------------------------------')

# g
y_pred = log_model.predict(X_test)
print(y_pred)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))
print('-------------------------------------------------------------------')


