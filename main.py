# -----------------------------------------------------------------------IMPORTING LIBRARIES----------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------READING PARKINSONS DATASET FROM CSV---------------------------------------------------------------

parkinsons_Dataset= pd.read_csv(r'C:\Users\piyan\Downloads\parkinsons.csv')
#print(parkinsons_Dataset.shape)
#print(parkinsons_Dataset.info())
#print(parkinsons_Dataset.describe())
#print(parkinsons_Dataset['status'].value_counts())
#print(parkinsons_Dataset.groupby('status').mean())

X = parkinsons_Dataset.drop(columns=['name','status'], axis =1)
Y = parkinsons_Dataset['status']


# -----------------------------------------------------------------TRAINING AND TESTING PART----------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
#print(X_test.shape)


scaler = StandardScaler()

scaler.fit(X_train)


X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------CREATING MODEL PART------------------------------------------------------------------------


PD_model = svm.SVC(kernel='linear')


# ---------------------------------------------------------------------TRAINING MODEL-----------------------------------------------------------------------------

PD_model.fit(X_train, Y_train)

# ----------------------------------------------------------------------TRAINING ACCURACY-------------------------------------------------------------------------

Prediction_of_X_Train = PD_model.predict(X_train)
Accuracy_of_Training_Set = accuracy_score(Y_train, Prediction_of_X_Train)

# -------------------------------------------------------------------TESTING ACCURACY-----------------------------------------------------------------------------

Prediction_of_X_Test = PD_model.predict(X_test)
Accuracy_of_test_Data = accuracy_score(Y_test, Prediction_of_X_Test)


# --------------------------------------------------------------------GETTING INPUTS FOR PREDICTION---------------------------------------------------------------

Input_Data_For_Prediction = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)


# ---------------------------------------------------------CONVERTING TO ARRAY FOR BETTER ARRANGEMENT OF DATA-----------------------------------------------------

Input_Data = np.asarray((Input_Data_For_Prediction))


# ----------------------------------------------------------RESHAPING THE DATA FRO BETTER PLACEMENTS OF DATA------------------------------------------------------

Reshaping_Input_Data = Input_Data.reshape(1,-1)

# -------------------------------------------------------------------STANDARDISING FOR ACCURATE O/P---------------------------------------------------------------

Standard_Data = scaler.transform(Reshaping_Input_Data)

# ----------------------------------------------------------------------------PRDEICTION PART---------------------------------------------------------------------

prediction = PD_model.predict(Standard_Data)
print(prediction)

# if prediction = 0:
# print("The person Do not have any parkinsons Disease")
# else:
# print("The person has Parkinsons Disease"