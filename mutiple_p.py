#Dạng 2. Phân lớp nhiều nhãn
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
filename='iris.csv'
#2a (2đ). Đọc vào file Iris.csv được biến dataframe df. Hiển thị df.
print('------------------------Câu 1-----------------------------')
df = pd.read_csv(filename)
print(df)
#2b (2đ). Sử dụng pháp chuyển đổi Label encoding, tạo cột mới  Species_number của df để chuyển cột 
#Species   gía trị chữ thành giá trị số. Hiển thị df.
print('------------------------Câu 2-----------------------------')
data_Species = df['Species'].unique()
print(data_Species)
df['Species']= label_encoder.fit_transform(df['Species'])
data_Species = df['Species'].unique()
df = df.rename(columns=({'Species':'Species_number'}))
print(data_Species)
print(df)
#2c (2đ). Từ dataframe df, trích chọn dataframe X gồm các cột SepalLengthCm, SepalWidthCm,PetalLengthCm,PetalWidthCm,
    #giá trị nhãn y là cột Species_number, hiển thị X,y.
print('------------------------Câu 3-----------------------------')
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
print(X)

Y = df[['Species_number']]
print(Y)
#2d (1đ). Chia ngẫu nhiên X,y thành X_train,y_train và X_test,y_test theo tỉ lệ 80:20. Hiển thị X_test,y_test.
print('------------------------Câu 4-----------------------------')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print('Shape X_train : ', X_train.shape)
print('Shape X_test : ', X_test.shape)
print('Shape y_train : ', y_train.shape)
print('Shape y_test : ', y_test.shape)
#2e (2đ). Sử dụng một trong các kỹ thuật phân lớp sau:
    #KNN, Logistics Regression, SVM, Decision tree, Random forest, XgBoost để huấn luyện mô hình học máy với tập dữ liệu X_train,y_train.
print('------------------------Câu 5-----------------------------')
model_lf = LogisticRegression()
model_lf.fit(X_train, y_train)
y_pred = model_lf.predict(X_test)
#2g (1đ). Dự báo kết quả và đánh giá độ chính xác dự báo trên tập X_test,y_test.
             #và dự báo kết quả nhãn của chỉ một mẫu dữ liệu mới với 4 thành phần SepalLengthCm, SepalWidthCm,PetalLengthCm,PetalWidthCm như sau: x=[4.5, 3.0, 1.4, 0.25]
print('------------------------Câu 6-----------------------------')
Matrix = confusion_matrix(y_test, y_pred)
print(Matrix)

evaluate_score = cross_val_score(estimator=model_lf, X = X_test, y = y_test)
print(evaluate_score)
