#importing the libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix , classification_report
import pickle

#Loading the data
train = pd.read_csv("train_data_evaluation_part_2.csv", index_col=False)
test = pd.read_csv("test_data_evaluation_part2.csv", index_col=False)

#Combining the data
df = pd.concat([train,test],axis=0)

#Getting unique values of the object type column
def unique_obj_col_value(df):
  for column in df:
    if df[column].dtype == 'object':
      print(f'{column}: {df[column].unique()}')
unique_obj_col_value(df)

for col in df.columns:
  print(f"{col}: {df[col].unique()}")

#I am droping the ID, Unnamed: 0 and Nationality since they are not going to add any value to the prediction and it's also a hinderance when it comes to deployment.
df.drop(['Unnamed: 0', 'ID','Nationality'], axis=1, inplace=True)

#Filling the missing Age column values with mean Age.
df['Age'].fillna(np.mean(df['Age']),inplace=True)

#This is our target col. Since this is a binary classification/prediction we can just consider anything greater than 0 as being checked in.
df['BookingsCheckedIn'] = df['BookingsCheckedIn'].replace([ 3,  1, 9,  2, 11, 12,  7,  8,  5,  6,  4, 66, 15, 29, 25, 10,
       17, 13, 26, 23, 57, 40, 18, 14, 24, 19, 20, 34], 1)

#Remaining columns with such values are also replaced with 1
df['BookingsNoShowed'] = df['BookingsNoShowed'].replace([2, 3], 1)
df['BookingsCanceled'] = df['BookingsCanceled'].replace([3,2, 4, 9], 1)

# Label encoding the object columns and then dumping them into a pickle file to use in the app.py
le = LabelEncoder()
df['DistributionChannel'] = le.fit_transform(df['DistributionChannel'])
pickle.dump(le, open('transform1.pkl', 'wb'))
df['MarketSegment'] = le.fit_transform(df['MarketSegment'])
pickle.dump(le, open('transform2.pkl', 'wb'))

#Dividing the data 
X = df.drop('BookingsCheckedIn', axis=1)
y = df['BookingsCheckedIn']

for col in df.columns:
  print(f"{col}: {df[col].unique()}")

#Scaling the columns, since most of the columns are having the values 0 and 1 I am going to scale the remaining to be somewhere near 0 and 1.
cols_to_scale = ['Age','DaysSinceCreation', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue','DaysSinceLastStay', 'DaysSinceFirstStay']
scaler = MinMaxScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
# pickle.dump(scaler, open('scale.pkl', 'wb'))

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

#Modeling (NLP)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

#Predicting
yp = clf.predict(X_test)
yp[:5]

#Coverting anything greater that 0.5 as 1 to make it clean for processing.
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

#Classification report of the model
print(classification_report(y_test,y_pred))

#Dumping the model to be used in the app.py
pickle.dump(clf, open('model1.pkl', 'wb'))
model = pickle.load(open('model1.pkl', 'rb'))

