import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#Dạng 1. Phân lớp nhị phân.
#1a. Đọc vào file user_data.csv được biến dataframe df. Hiển thị df.
print("Cau 1:")
df_dataframe=pd.read_csv("User_Data.csv")

#1b (2đ). Sử dụng pháp chuyển đổi Label encoding, tạo cột mới Gender_number của df để chuyển cột Gender gía trị chữ thành giá trị số. Hiển thị df.
print("Cau 2:")
df_dataframe["Gender_number"] = preprocessing.LabelEncoder().fit_transform(df_dataframe["Gender"])
print(df_dataframe)

#1c. Từ dataframe df, trích chọn dataframe X gồm các cột Gender_numer,Age,EstimatedSalary
#giá trị nhãn y là cột Purchased, hiển thị X,y.
print("Cau 3:")
X = df_dataframe[['Gender_number','Age','EstimatedSalary']]
print("X:\n",X)
y = df_dataframe[['Purchased']]
print("\ny:\n",y)

#1d (1đ). Chia ngẫu nhiên X,y thành X_train,y_train và X_test,y_test theo tỉ lệ 80:20. 
#Hiển thị X_test,y_test.
print("Cau 4:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#1e (2đ). Sử dụng một trong các kỹ thuật phân lớp nhị phân sau:
  #Logistics Regression, SVM, Adaboost để huấn luyện mô hình học máy với tập dữ liệu X_train,y_train.
print("Cau 5:")
model_lf = LogisticRegression()
model_lf.fit(X_train, y_train)
y_pred = model_lf.predict(X_test)

#1g (1đ). Dự báo kết quả và đánh giá độ chính xác dự báo trên tập X_test,y_test.
print("Cau 6:")
Matrix = confusion_matrix(y_test, y_pred)
print(Matrix)

evaluate_score = cross_val_score(estimator=model_lf, X = X_test, y = y_test)
print(evaluate_score)