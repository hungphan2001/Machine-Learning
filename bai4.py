from unittest import result
#pandas đọc file
import pandas as pd
# đồ thị 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#PCA 
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
# ma trận
from sklearn.metrics import confusion_matrix
# kỹ thuật học máy
from sklearn.decomposition import PCA
# train file csv
from sklearn.model_selection import train_test_split  
# kỹ thuật học máy
from sklearn.linear_model import LogisticRegression  
# xác định độ chính xác trung bình
from sklearn.metrics import accuracy_score 
# thuư viện làm việc với mảng 
import numpy as np
# 1a. Đọc file housing1.csv ghi vào frame df_housing
print("Cau 1:")
df_housing = pd.read_csv("housing1.csv")
print (df_housing)

# 2a In ra 10 dòng đầu tiên
print("Cau 2:")
df_housing = pd.read_csv("housing1.csv")
results = df_housing.head(10)
print(results)

#3. Gán X là tất cả các cột có giá trị số của df_house (trừ cột cuối cùng). Hiện thị 10 dòng đầu tiên của X.
print("Cau 3:")
X = df_housing.select_dtypes(include=np.number)
print('10 First line of X :\n', X.head(10))
print('=======================================================================================================================')

#4. Gán y là cột cuối cùng của df_house. Hiện thị 10 dòng đầu tiên của y.
print("Cau 4:")
Y = df_housing ['Address']
result = Y.head(10)
print("10 dong dau cua y : \n",result)
print('=======================================================================================================================')

#5. Giảm X thành 1 chiều bằng kỹ thuật PCA. Hiện thị 10 dòng đầu tiên của X.
print("cau 5:")
pca_breast = PCA(n_components=1)
X_new = pd.DataFrame(data = pca_breast.fit_transform(X))
print(X_new.head(10))
print('=======================================================================================================================')

#6. Vẽ đồ thị với X, y (số lượng 50 điểm dữ liệu)   
print("Cau 6:")
plt.plot(X_new[0][:50], Y[:50], 'o')
plt.show()
print('=======================================================================================================================')

#7. Chia tập (X, y) thành (X_train, y_train) và (X_test, y_test) theo tỉ lệ 50-50.
print("Cau 7:")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
print('=======================================================================================================================')

#8. Tạo mô hình học model_lr: huấn luyện tập X_train với kỹ thuật học máy linear regression.
print("Cau 8:")
model= LinearRegression().fit(x_train, y_train)
print('=======================================================================================================================')

#9. Hiện thị các hệ số sau khi huấn luyện (coef_) 
#10. Hiện thị điểm dữ liệu X5 (dòng số 5 của X) và nhãn y5 tương ứng.
#11. Dự đoán y_pred5 của điểm dữ liệu X5. So sánh kế quả với 10.
#12. Dựa vào 9, tính y_ped5. So sánh với 11.
#13. Tính sai số mean_squared_error cho điểm y_pred5.
#14. Dự đoán kết quả của model_lr với tập X_test thu được y_pred_lf. Hiện thị 50 kết quả dự đoán đầu tiên.
#15. Hiển thị mean_squared_error, r2_score của model_lr trên tập X_test.
#16. Dựa vào các hệ số trong câu 9. Vẽ đồ thị hiện thị đường dự đoán.