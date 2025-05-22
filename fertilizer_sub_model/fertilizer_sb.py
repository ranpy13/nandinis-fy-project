from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import seaborn as sns

from utils.logger_util import setup_logger

logger = setup_logger(__name__)

df = pd.read_csv("FertilizerPrediction.csv")
logger.debug(df.shape)
logger.debug(df.size)
logger.debug(df.describe())

unique_values = df.apply(lambda x: len(x.unique()))

# check for categorical attributes
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col

# print the categorical columns
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()


#encoding Soil Type variable
encode_soil = LabelEncoder()
df['SoilType'] = encode_soil.fit_transform(df['SoilType'])

#creating the DataFrame
Soil_Type = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
Soil_Type = Soil_Type.set_index('Original')
Soil_Type

# encoding the crop type variables
encode_crop =  LabelEncoder()
df['CropType'] = encode_crop.fit_transform(df['CropType'])

#creating the DataFrame
Crop_Type = pd.DataFrame(zip(encode_crop.classes_,encode_crop.transform(encode_crop.classes_)),columns=['Original','Encoded'])
Crop_Type = Crop_Type.set_index('Original')
Crop_Type


# encoding fertilizer name type variable
encode_ferti = LabelEncoder()
df['FertilizerName'] = encode_ferti.fit_transform(df['FertilizerName'])

#creating the DataFrame
Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
Fertilizer = Fertilizer.set_index('Original')
Fertilizer

# it shows the count of each soil type 
plt.figure(figsize=(10,5))
sns.countplot(x='SoilType', data = df)

#it show the count of each crop type
plt.figure(figsize=(10,8))
sns.countplot(x='CropType', data = df)

plt.figure(figsize=(8,5))
sns.countplot(x='FertilizerName', data = df)

#correlation heatmap
plt.figure(figsize=[10,8])
sns.heatmap(df.corr(),annot=True)
plt.show()


## Model Development
x=df.drop(["FertilizerName"],axis=1)
y=df["FertilizerName"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.2)

# Feature Scaling
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

#evaluating KNN for Classification
acc=classifier.score(x_test,y_test)
print("Accuracy of testing set knn:",acc)

#evaluating KNN for Classification
acc=classifier.score(x_train,y_train)
print("Accuracy of training set knn:",acc)

# Calculating the Accuracy
from sklearn import metrics
y_pred=classifier.predict(x_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))

#classification report
from sklearn.metrics import classification_report
print("Classification report of KNN classifier:\n",classification_report (y_test,y_pred))



a=float(input("enter temparature value:"))
b=float(input("enter Humidity value:"))
c=float(input("enter Moisture value:"))
d=int(input("enter soil type value:"))
e=int(input("enter Crop type value:"))
f=float(input("enter Nitrogen value:"))
g=float(input("enter Potassium value:"))
h=float(input("enter Phoshporous value:"))
ans=classifier.predict([[a,b,c,d,e,f,g,h]])
if(((a>=20)&(a<=40)) & ((b>40)&(b<70)) & ((c>=20)&(c<=70)) & ((d>=0)&(d<=4)) & ((e>=0)&(e<=10)) & ((f>=0)&(f<=50)) & ((g>=0)&(g<=20)) & 
    ((h>=0)&(h<=50))):
    if ans[0] == 0:
        print("10-26-26")
    elif ans[0] ==1:
        print("14-35-14")
    elif ans[0] == 2:
        print("17-17-17")
    elif ans[0] == 3:
        print("20-20")
    elif ans[0] == 4:
        print("28-28")
    elif ans[0]==5:
        print("DAP")
    elif ans[0]==6:
        print("Urea")
else:
    print("invalid input")
    



#importing libraries
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
clf.fit(x_train, y_train)

print("score of traing set:",clf.score(x_train,y_train))
print("score of testing set:",clf.score(x_test,y_test))

y_pred=clf.predict(x_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

a=float(input("enter temparature value:"))
b=float(input("enter Humidity value:"))
c=float(input("enter Moisture value:"))
d=int(input("enter soil type value:"))
e=int(input("enter Crop type value:"))
f=float(input("enter Nitrogen value:"))
g=float(input("enter Potassium value:"))
h=float(input("enter Phoshporous value:"))
ans=clf.predict([[a,b,c,d,e,f,g,h]])
if(((a>=20)&(a<=40)) & ((b>40)&(b<70)) & ((c>=20)&(c<=70)) & ((d>=0)&(d<=4)) & ((e>=0)&(e<=11)) & ((f>=0)&(f<=50)) & ((g>=0)&(g<=20)) & 
   ((h>=0)&(h<=50))):
    if ans[0] == 0:
        print("10-26-26")
    elif ans[0] ==1:
        print("14-35-14")
    elif ans[0] == 2:
        print("17-17-17")
    elif ans[0] == 3:
        print("20-20")
    elif ans[0] == 4:
        print("28-28")
    elif ans[0]==5:
        print("DAP")
    else:
        print("Urea")
else:
    print("Input value is in valid")