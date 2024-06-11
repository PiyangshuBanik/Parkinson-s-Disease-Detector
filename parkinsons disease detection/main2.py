import tkinter as tk
from tkinter import Entry, Label, Button
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Reading Parkinsons dataset from CSV
parkinsons_Dataset = pd.read_csv(r'C:\Users\piyan\Downloads\parkinsons.csv')

X = parkinsons_Dataset.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_Dataset['status']

# Training and testing part
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creating model
PD_model = svm.SVC(kernel='linear')

# Training model
PD_model.fit(X_train, Y_train)

# Training accuracy
Prediction_of_X_Train = PD_model.predict(X_train)
Accuracy_of_Training_Set = accuracy_score(Y_train, Prediction_of_X_Train)

# Testing accuracy
Prediction_of_X_Test = PD_model.predict(X_test)
Accuracy_of_test_Data = accuracy_score(Y_test, Prediction_of_X_Test)

def reset_values():
    for entry in input_entries:
        entry.delete(0, 'end')  # Clear the values in the entry fields
    result_label.config(text="")  # Clear the result label

def predict_parkinsons():
    # Get the input data from the user
    input_data = [float(entry.get()) for entry in input_entries]
    
    # Standardize the input data
    standard_data = scaler.transform([input_data])
    
    # Make a prediction
    prediction = PD_model.predict(standard_data)

    # Determine the message based on the prediction
    if prediction[0] == 0:
        message = "You may not have Parkinson's Disease"
    else:
        message = "You may have Parkinson's Disease"
    
    # Calculate accuracy on the test data and convert it to a percentage
    test_predictions = PD_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_predictions) * 100
    
    # Display the prediction result and accuracy as a percentage
    result_label.config(text=f"Prediction: {prediction[0]}, Test1 Accuracy: {test_accuracy:.0f}%")


# Create a tkinter window
window = tk.Tk()
window.title("Parkinson's Prediction")

# Create labels and entry fields for input
input_labels = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
                "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2",
                "D2", "PPE"]

input_entries = []

for i, label in enumerate(input_labels):
    label = Label(window, text=label)
    label.grid(row=i, column=0)
    entry = Entry(window)
    entry.grid(row=i, column=1)
    input_entries.append(entry)

# Create a button for prediction
predict_button = Button(window, text="Predict Parkinson's", command=predict_parkinsons)
predict_button.grid(row=len(input_labels), column=0)

# Create a button for resetting values
reset_button = Button(window, text="Reset Values", command=reset_values)
reset_button.grid(row=len(input_labels), column=1)

# Create a label for displaying the prediction result
result_label = Label(window, text="")
result_label.grid(row=len(input_labels) + 1, columnspan=2)

# Start the GUI application
window.mainloop()