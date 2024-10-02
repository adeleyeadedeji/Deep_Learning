# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:49:13 2023

@author: GREAT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt


# Load the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('extracted_data.csv')

# Data Preprocessing
# Data Preprocessing
# Encode categorical variables
label_encoders = {}
categorical_columns = ['Category', 'Content Rating']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Handle non-numeric values in numerical columns
numerical_columns = ['Installs', 'Size']

# Replace non-numeric values with NaN
for column in numerical_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Drop rows with missing values
data = data.dropna()

# Normalize numerical features
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Split the data into train and test sets
X = data.drop(columns=['Rating'])
y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Recurrent Neural Network (RNN)
model_rnn = Sequential()
model_rnn.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model_rnn.add(Dense(1))
model_rnn.compile(optimizer='adam', loss='mse')

# Record training history
history_rnn = model_rnn.fit(
    tf.convert_to_tensor(X_train.values.reshape(-1, X_train.shape[1], 1), dtype=tf.float32), 
    tf.convert_to_tensor(y_train, dtype=tf.float32), 
    epochs=10, 
    batch_size=32,
    validation_data=(
        tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32), 
                     tf.convert_to_tensor(y_test, dtype=tf.float32)
    )
)

# Convolutional Neural Network (CNN)
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(1))
model_cnn.compile(optimizer='adam', loss='mse')

# Record training history for CNN
history_cnn = model_cnn.fit(
    tf.convert_to_tensor(X_train.values.reshape(-1, X_train.shape[1], 1), dtype=tf.float32), 
    tf.convert_to_tensor(y_train, dtype=tf.float32), 
    epochs=10, 
    batch_size=32,
    validation_data=(
        tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32), 
                     tf.convert_to_tensor(y_test, dtype=tf.float32)
                     )
)


# Model Evaluation
# Make predictions using both models
y_pred_rnn = model_rnn.predict(tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32))
y_pred_cnn = model_cnn.predict(tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32))

# Assuming you have already trained and made predictions with the RNN and CNN models
# y_pred_rnn and y_pred_cnn contain the predicted values

# Plot the true values (y_test) against the predicted values for RNN
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rnn, color='blue', label='RNN Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('RNN Model: True vs. Predicted Ratings')
plt.legend()
plt.grid(True)
plt.show()

# Plot the true values (y_test) against the predicted values for CNN
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_cnn, color='red', label='CNN Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('CNN Model: True vs. Predicted Ratings')
plt.legend()
plt.grid(True)
plt.show()


# Calculate MSE and RMSE for both models
mse_rnn = mean_squared_error(y_test, y_pred_rnn)
rmse_rnn = sqrt(mse_rnn)
mse_cnn = mean_squared_error(y_test, y_pred_cnn)
rmse_cnn = sqrt(mse_cnn)


# Create a bar chart to visualize MSE and RMSE for both models
models = ['RNN', 'CNN']
mse_values = [mse_rnn, mse_cnn]
rmse_values = [rmse_rnn, rmse_cnn]

index = np.arange(len(models))

# Plot MSE
plt.figure(figsize=(8, 6))
plt.bar(index, mse_values, tick_label=models, color='skyblue')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('MSE Comparison for RNN and CNN Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot RMSE
plt.figure(figsize=(10, 6))
plt.bar(index, rmse_values, tick_label=models, color='lightcoral')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('RMSE Comparison for RNN and CNN Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Model Selection
if mse_rnn < mse_cnn:
    selected_model = model_rnn
    selected_model_type = "RNN"
else:
    selected_model = model_cnn
    selected_model_type = "CNN"

selected_model.save(f"{selected_model_type.lower()}_model.h5")

print(f"Selected Model: {selected_model_type}")
print(f"MSE ({selected_model_type}): {mse_rnn if selected_model_type == 'RNN' else mse_cnn}")
print(f"RMSE ({selected_model_type}): {rmse_rnn if selected_model_type == 'RNN' else rmse_cnn}")


# More Useful Charts

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Calculate R-squared (optional, for model evaluation)
r2_rnn = r2_score(y_test, y_pred_rnn)
r2_cnn = r2_score(y_test, y_pred_cnn)

# Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model_rnn.history.history['loss'], label='RNN Training Loss', color='blue')
plt.plot(model_rnn.history.history['val_loss'], label='RNN Validation Loss', color='red')
plt.title('RNN Model Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model_cnn.history.history['loss'], label='CNN Training Loss', color='blue')
plt.plot(model_cnn.history.history['val_loss'], label='CNN Validation Loss', color='red')
plt.title('CNN Model Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Actual vs. Predicted Scatter Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rnn, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray')
plt.title('RNN Model: Actual vs. Predicted Ratings')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_cnn, color='red')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray')
plt.title('CNN Model: Actual vs. Predicted Ratings')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# Residual Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
residuals_rnn = y_test - model_rnn.predict(tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32)).flatten()
plt.scatter(model_rnn.predict(tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32)).flatten(), residuals_rnn, color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.title('RNN Model: Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
residuals_cnn = y_test - model_cnn.predict(tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32)).flatten()
plt.scatter(model_cnn.predict(tf.convert_to_tensor(X_test.values.reshape(-1, X_test.shape[1], 1), dtype=tf.float32)).flatten(), residuals_cnn, color='red')
plt.axhline(0, color='gray', linestyle='--')
plt.title('CNN Model: Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

# Histogram of Residuals
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(y_test - y_pred_rnn, bins=30, color='blue', kde=True)
plt.title('RNN Model: Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_cnn, bins=30, color='red', kde=True)
plt.title('CNN Model: Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Print R-squared values
print(f'R-squared (RNN): {r2_rnn:.2f}')
print(f'R-squared (CNN): {r2_cnn:.2f}')


# # AI Application Integration
# # Assuming you have a function `predict_app_rating` that takes new app attributes and returns a rating prediction
# def predict_app_rating(new_attributes):
#     # Preprocess new_attributes (encode categorical and normalize numerical features)
#     new_attributes = pd.DataFrame(new_attributes, columns=X.columns)
#     for column in categorical_columns:
#         new_attributes[column] = label_encoders[column].transform(new_attributes[column])
#     new_attributes[numerical_columns] = scaler.transform(new_attributes[numerical_columns])
    
#     # Make a prediction using the selected model
#     prediction = selected_model.predict(tf.convert_to_tensor(new_attributes.values.reshape(1, -1, 1), dtype=tf.float32))
    
#     return prediction[0][0]

# # Example usage
# new_app_attributes = {
#     'Category': ['Adventure'],
#     'Installs': [1000],
#     'Free': [True],
#     'Price': [0],
#     'Size': [5.0],
#     'Content Rating': ['Everyone'],
#     'Ad Supported': [False],
#     'In App Purchases': [True],
#     'Editors Choice': [False]
# }

# predicted_rating = predict_app_rating(new_app_attributes)
# print(f"Predicted Rating: {predicted_rating}")

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the selected model (RNN or CNN)
selected_model_type = "RNN"
selected_model = load_model("rnn_model.h5")

# Load label encoders and scalers used for preprocessing
label_encoders = {}
scaler = MinMaxScaler()
# Load label encoders and scaler from saved files (e.g., label_encoders.pkl, scaler.pkl)

# Function to preprocess user input and make predictions
def predict_app_rating():
    try:
        # Get user input from the entry widgets
        category = category_entry.get()
        installs = float(installs_entry.get())
        free = bool(free_var.get())
        price = float(price_entry.get())
        size = float(size_entry.get())
        content_rating = content_rating_combobox.get()
        ad_supported = bool(ad_supported_var.get())
        in_app_purchases = bool(in_app_purchases_var.get())
        editors_choice = bool(editors_choice_var.get())

        # Create a DataFrame for preprocessing
        input_data = pd.DataFrame({
            'Category': [category],
            'Installs': [installs],
            'Free': [free],
            'Price': [price],
            'Size': [size],
            'Content Rating': [content_rating],
            'Ad Supported': [ad_supported],
            'In App Purchases': [in_app_purchases],
            'Editors Choice': [editors_choice]
        })

        # Encode categorical variables and normalize numerical features
        for column in input_data.columns:
            if column in label_encoders:
                input_data[column] = label_encoders[column].transform(input_data[column])
        input_data[['Installs', 'Size']] = scaler.transform(input_data[['Installs', 'Size']])

        # Make a prediction using the selected model
        prediction = selected_model.predict(tf.convert_to_tensor(input_data.values.reshape(1, -1, 1), dtype=tf.float32))

        # Display the predicted rating to the user
        predicted_rating_label.config(text=f"Predicted Rating: {prediction[0][0]:.2f}")
    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Create the main application window
app = tk.Tk()
app.title("App Rating Prediction")

# Create and configure input widgets
category_label = ttk.Label(app, text="Category:")
category_label.grid(row=0, column=0, padx=10, pady=5, sticky="E")
category_entry = ttk.Entry(app)
category_entry.grid(row=0, column=1, padx=10, pady=5)

installs_label = ttk.Label(app, text="Installs:")
installs_label.grid(row=1, column=0, padx=10, pady=5, sticky="E")
installs_entry = ttk.Entry(app)
installs_entry.grid(row=1, column=1, padx=10, pady=5)

free_label = ttk.Label(app, text="Free:")
free_label.grid(row=2, column=0, padx=10, pady=5, sticky="E")
free_var = tk.BooleanVar()
free_checkbox = ttk.Checkbutton(app, variable=free_var)
free_checkbox.grid(row=2, column=1, padx=10, pady=5)

# Add more input widgets for other features...

# Create a button to make predictions
predict_button = ttk.Button(app, text="Predict", command=predict_app_rating)
predict_button.grid(row=3, column=0, columnspan=2, pady=10)

# Display the predicted rating
predicted_rating_label = ttk.Label(app, text="", font=("Helvetica", 16))
predicted_rating_label.grid(row=4, column=0, columnspan=2, pady=10)

# Start the Tkinter main loop
app.mainloop()

