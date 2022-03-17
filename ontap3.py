# Dạng 4. Hồi quy với phép hồi quy tuyến tính.
# 1a (2đ). Đọc vào file housing.csv được biến dataframe df. Hiển thị df.
# 1b (2đ). Từ dataframe df, trích chọn dataframe X gồm 5 cột:
# Avg. Area Income, Avg. Area House Age,Avg. Area Number of Rooms,Avg. Area Number of Bedrooms,Area Population,
# giá trị nhãn y là cột Price, hiển thị X,y.
# 1c (1đ). Chia ngẫu nhiên X,y thành X_train,y_train và X_test,y_test theo tỉ lệ 80:20. Hiển thị X_test,y_test.
# 1d (2đ). Sử dụng kỹ thuật hồi quy tuyến tính để huấn luyện mô hình hồi quy với tập dữ liệu X_train,y_train.
# 1e (2đ). Dự báo kết quả và đánh giá độ chính xác dự báo trên tập X_test,y_test.
# 1g (1đ). Ước lượng giá y của căn hộ khi biết các giá trị
# 5 thành phần của căn hộ Avg. Area Income, Avg. Area House Age,Avg. Area Number of Rooms,Avg. Area Number of Bedrooms,Area Population
# được cho như sau: x=[61200.06718,5.60588984,7.51272743,5.13,35882.1594]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics

file_name = 'housing.csv'

# a
df = pd.read_csv(file_name)
# print(df)
print('-------------------------------------------------------------------')

# b
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
# print(X)
y = df[['Price']]
# print(y)
print('-------------------------------------------------------------------')

# c
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print('X_test')
# print(X_test.shape)
# print(X_test)
# print('y_test')
# print(y_test.shape)
# print(y_test)
print('-------------------------------------------------------------------')

# d
lng_model = linear_model.LinearRegression(fit_intercept=False)
lng_model = lng_model.fit(X_train ,y_train)
# print the coefficients
print(lng_model.intercept_)
print(lng_model.coef_)
print('-------------------------------------------------------------------')

# e
y_pred = lng_model.predict(X_test)
# print(y_pred)
# print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('-------------------------------------------------------------------')

# g
x=[[61200.06718, 5.60588984, 7.51272743, 5.13, 35882.1594]]
y_pred = lng_model.predict(x)
print(y_pred)
print('-------------------------------------------------------------------')
