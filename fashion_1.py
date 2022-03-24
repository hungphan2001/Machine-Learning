from matplotlib.pyplot import axis
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
#1a. Đọc vào file fashion.csv được biến dataframe df. Hiển thị df.
print("Cau 1:")
df_dataframe=pd.read_csv("fashion.csv")
print(df_dataframe)

#1b (2đ). chọn giá trị nhãn y là cột đầu tiên, và dataframe X là tất cả các cột còn lại. 
print("Cau 2:")
X = df_dataframe.drop(['label'], axis=1)
y = df_dataframe['label']
print(X)
print(y)

#1c. Đếm và hiển thị số lượng các mẫu dữ liệu của từng giá trị nhãn y. Chia ngẫu nhiên X,y thành X_train,y_train và X_test,y_test theo tỉ lệ 60:40. Hiển thị X_test,y_test.
print("Cau 3:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(X_test)
print(y_test)

#1d (1đ). Sử dụng một trong các kỹ thuật phân lớp để huấn luyện mô hình học máy với tập dữ liệu X_train,y_train.
print("Cau 4:")
log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(log_model.coef_)
y_pred = log_model.predict(X_test)

#1e (2đ). dự đoán kết quả và đánh giá độ chính xác dự đoán trên tập X_test,y_test với Accuracy, Recall, Precisionprint("Cau 5:")
print("Cau 5:")
y_pred = log_model.predict(X_test)
print("Y_pred = ",y_pred)
print("Accuracy score:",accuracy_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred,average='micro'))
print('Precision',precision_score(y_test,y_pred,average='micro'))


#1g (1đ). Dự đoán nhãn của dòng thứ 100 của fashion_df và hiển thị kết quả dự đoán. Chuyển dữ liệu dòng thứ 100 của fashion_df thành một ma trận ảnh gray I, kích thước 28x28. Sử dụng OpenCV để hiển thị ảnh I.print("Cau 6:")
print("Cau 6:")
X4 = X.iloc[4,:]
print(X4)
X4 = X4.to_numpy().reshape((28,28))

cv2.imshow("anh",X4.astype(np.uint8))
cv2.waitKey(0)