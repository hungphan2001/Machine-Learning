import pandas as pd 

#Đọc file Real estate.csv vào  real_estate_df. 
estate_data = pd.read_csv("Real estate.csv")
# print(fashion_data.shape)

# estate_data = estate_data.sample(frac = 1)
print(estate_data)
print(estate_data.columns)

X = estate_data.drop(["No","Y house price of unit area"], axis = 1)
print(X)
y = estate_data["Y house price of unit area"]
print(y)

# Chia ngẫu nhiên X,y thành X_train,y_train và X_test,y_test theo tỉ lệ 50:50.
#Sử dụng kỹ thuật hồi quy Linear Regression để huấn luyện với X_train,y_train.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# # print(X_train.shape)

from sklearn.linear_model import LinearRegression
lr_model= LinearRegression().fit(X_train, y_train)
print(lr_model.coef_)

y_pred = lr_model.predict(X_test)

# # print(y_pred)
# # print(y_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

X5 = X.iloc[5,:]
y5 = y.iloc[5]
print(X5)
print(y5)
y_pred5 = lr_model.predict([X5])
print(y_pred5)
#tính mse 
print("Bình phương sai số bằng hàm", mean_squared_error([y5],y_pred5))
print("Bình phương sai số tính tay ",(y5-y_pred5)*(y5-y_pred5))

# print(clf.predict([[2013.58300, 35.90000, 616.40040, 3.00000, 24.97723, 121.53767]]))
# print(mean_squared_error([36.8], clf.predict([[2013.58300, 35.90000, 616.40040, 3.00000, 24.97723, 121.53767]])))



# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
# y_pred = neigh.predict(X_test)
# print(accuracy_score(y_test, y_pred))


# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=19).fit(X_train)

# number = X.iloc[3,:].to_numpy().reshape((28,28))
# print(number)

# import cv2
# import numpy as np
# cv2.imshow("anh",number.astype(np.uint8))

# cv2.waitKey(0) 
  
# #closing all open windows 
# cv2.destroyAllWindows()


