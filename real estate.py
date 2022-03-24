import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1 Đọc file Real estate.csv vào  real_estate_df. 
real_estate_df = pd.read_csv('Real estate.csv')
print(real_estate_df)

# 2 loại bỏ cột đầu tiên, gán y là cột cuối cùng và X là các cột còn lại. 
real_estate_df = real_estate_df.drop(["No"],axis = 1)
X = real_estate_df.drop(["Y house price of unit area"],axis = 1)
y = real_estate_df["Y house price of unit area"]

# 3 Chia ngẫu nhiên X,y thành X_train,y_train và X_test,y_test theo tỉ lệ 50:50.
# Sử dụng kỹ thuật hồi quy Linear Regression để huấn luyện với X_train,y_train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
model_lr = LinearRegression().fit(X_train, y_train)

# 4 Dự đoán trên tập X_test 
y_pred = model_lr.predict(X_test)
print(y_pred)

# 5 Tính  mse 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

# 6 (0.5đ). Dự đoán kết quả và đánh giá độ chính xác dự đoán trên một điểm dữ liệu (X57,y57) với:
# X57= [2013.58300, 35.90000, 616.40040, 3.00000, 24.97723, 121.53767] và y57 = 36.8
X57 = [2013.58300, 35.90000, 616.40040, 3.00000, 24.97723, 121.53767] 
y57 = 36.8
X57_new = pd.DataFrame([X57], 
             columns=['X1 transaction date',  'X2 house age',  'X3 distance to the nearest MRT station',  'X4 number of convenience stores', 'X5 latitude', 'X6 longitude'])
#y57_new =  pd.DataFrame([y57], columns=['Y house price of unit area'])
#y_pred57 = model_lr.predict([X57])
y_pred57 = model_lr.predict(X57_new)
print(y_pred57)
# print('Kiểm tra độ chính xác')
# r2_score = model_lr.score(X57_new,y57_new)
# r2_score = model_lr.score(X_test,y_test)
# print("Độ chính xác của hồi quy tuyến tính: ",r2_score*100,'%')