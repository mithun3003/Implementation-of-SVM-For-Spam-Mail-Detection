# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.
```

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MITHUN S
RegisterNumber: 24901037
*/
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/b3531e9a-5b85-4808-ade1-fa0f3c0927e2)

![SVM For Spam Mail Detection](![image](https://github.com/user-attachments/assets/222ca4e7-faef-495d-8ee3-02cef9146db9)
)
![image](https://github.com/user-attachments/assets/464bc1a2-b943-4bec-aa30-afb0b07bac49)
![image](https://github.com/user-attachments/assets/4b75f981-f70e-4cae-a2f7-04bf4438f519)
![image](https://github.com/user-attachments/assets/84cbf434-b5fb-407a-a62a-6da53310c760)
![image](https://github.com/user-attachments/assets/53f3e5ca-e940-4629-831c-5c038ff96f94)
![image](https://github.com/user-attachments/assets/ff6b763f-d413-4c9c-a868-5ef320eac1db)







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
