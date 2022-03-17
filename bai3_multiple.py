import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1a. Đọc file bank.csv ghi vào frame iris
iris = pd.read_csv("iris.csv")
iris = iris.drop(["Id"],axis = 1)
# ls_conditions = ['Iris-setosa', 'Iris-versicolor']
# iris = iris[iris['Species'].isin(ls_conditions)]
# print(iris)

#----------------------------- 
# 1b. In ra kích thước và 10 dòng đầu tiên của iris
result = iris.head(10)
column_number = iris.shape[1]
row_number = iris.shape[0]
# print('column:',column_number,'row:',row_number)
# print('10 dong dau:')
# print(result)

#----------------------------- 
# 1c. Gán X là tất cả các cột có giá trị số của iris. Hiện thị 10 dòng đầu tiên của X
iris_X = iris.drop(["Species"],axis = 1)
X = iris_X.select_dtypes(include="number")
# print(X.head(10))

#----------------------------- 
# 1d. Gán y là cột cuối cùng của iris. Hiện thị 10 dòng đầu tiên của y
y = iris['Species']
# print(y.head(10))

#----------------------------- 
# 1e. Chia tập (X, y) thành (X_train, y_train) và (X_test, y_test) theo tỉ lệ 50-50
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# ----------------------------
#1f. In ra kích thước của X_train, y_train và X_test, y_test
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# ----------------------------
#1g. Tạo mô hình học model_lf: huấn luyện tập (X_train, y_train) với kỹ thuật học máy LogisticRegression
model_lf = KNeighborsClassifier()
# model_lf = OneVsOneClassifier(model_lf)
model_lf = model_lf.fit(x_train, y_train)
y_pred = model_lf.predict(x_test)
print(y_pred)

# ----------------------------
#1h. Hiện thị kết quả Accuracy Score của model_lf
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", acc_score)

# ----------------------------
#1n. In ra classification report của model_lf --> precision, recall
print(classification_report(y_test,y_pred))

# ----------------------------
#1o. In ra confusion matrix của model_lf --> đếm mẫu dự đoán đúng/sai
confusion_matrix1 = confusion_matrix(y_test,y_pred)
# print(confusion_matrix1)
matrix_list = []
for i in confusion_matrix1:
    for j in i:
        matrix_list.append(j)
# ----------------------------
#1p. Tính accuracy, precision và recall cho 2 lớp (no/yes) dựa vào confusion matrix ở 1o. So sánh kết quả với 1n và 1h.
# no
# accuracy_cal = (matrix_list[0]+matrix_list[3])/(matrix_list[0]+matrix_list[1]+matrix_list[2]+matrix_list[3])
# print(accuracy_cal)
# precision_cal = (matrix_list[0])/(matrix_list[0]+matrix_list[2])
# print(precision_cal)
# recall_cal = (matrix_list[0])/(matrix_list[0]+matrix_list[1])
# print(recall_cal)

# yes
# precision_cal = (matrix_list[3])/(matrix_list[1]+matrix_list[3])
# print(precision_cal)
# recall_cal = (matrix_list[3])/(matrix_list[2]+matrix_list[3])
# print(recall_cal)
# ----------------------------
#1i. Viết hàm tính kết quả Accuracy Score. So sánh kết quả với 1h.
y_test_value = list(y_test)
cnt = 0
for i in range(y_test.shape[0]):
    if y_test_value[i] == y_pred[i]:
        cnt+=1
acc_score_test = cnt/y_test.shape[0]
# if acc_score == acc_score_test:
#     print('acc_score==acc_score_test')
# elif acc_score > acc_score_test:
#     print('acc_score>acc_score_test')
# else:
#     print('acc_score<acc_score_test')

# ----------------------------
#1j. Chia lại tập (X, y) thành (X_train, y_train) và (X_test, y_test) theo tỉ lệ 80-20
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ----------------------------
#1k. Tạo mô hình học model_svm: huấn luyện tập (X_train, y_train) với kỹ thuật học máy SVM
model_svm = LinearSVC()
# model_svm = OneVsOneClassifier(model_svm)
model_svm = model_svm.fit(x_train, y_train)
y_pred = model_svm.predict(x_test)
# print(model_svm)
# w = model_svm.coef_
# b = model_svm.intercept_
# print('w = ', w)
# print('b = ', b)
# print(x_test)
# print(y_pred)

# ----------------------------
#1m. Hiện thị kết quả Accuracy Score của model_svm
acc_score = accuracy_score(y_test, y_pred)
# print(acc_score)
# print(classification_report(y_test,y_pred))

# ----------------------------
#1l. Viết hàm tính kết quả Accuracy Score. So sánh kết quả với 1m.
y_test_value = list(y_test)
cnt = 0
for i in range(y_test.shape[0]):
    if y_test_value[i] == y_pred[i]:
        cnt+=1
acc_score_test = cnt/y_test.shape[0]
# if acc_score == acc_score_test:
#     print('acc_score==acc_score_test')
# elif acc_score > acc_score_test:
#     print('acc_score>acc_score_test')
# else:
#     print('acc_score<acc_score_test')