#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import PySimpleGUI as sg
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor


# In[ ]:


def preprocess_data(file_path):
    
    label_encoders = {}  # Store label encoders for each column || use for decoding it (New Feature will add soon Predictting With GUI by PYSimpleGui library)
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format. Exiting.")
        exit()
    
    # Use for Check waht will do? ==>> drop NUlls OR Fill with Mean for Numeric columns and Mode for Categorical columnss
    def calculate_null_percentage(column): 
        total_rows = len(column)
        null_count = column.isnull().sum()
        return (null_count / total_rows) * 100
    
    def split_nulls(column):
        is_numeric = column.apply(lambda x: pd.api.types.is_numeric_dtype(x))
        numeric_nulls = column[is_numeric]
        string_nulls = column[~is_numeric]
        return numeric_nulls, string_nulls
    
    def fill_nulls(column):
        if pd.api.types.is_numeric_dtype(column):
            median_value = column.median()
            column.fillna(median_value, inplace=True)
        else:
            mode_value = column.mode()[0]
            column.fillna(mode_value, inplace=True)
    
    
    
    for column_name in df.columns:
        if df[column_name].dtype == 'object':
            converted_column = pd.to_numeric(df[column_name], errors='coerce')
            print(f"is {column_name} numerical string? {not converted_column.isna().all()}")
            if not converted_column.isna().all():
                df[column_name] = converted_column  
            else:
                label_encoder = LabelEncoder()
                encoded_values = label_encoder.fit_transform(df[column_name])
                df[column_name] = encoded_values
                label_encoders[column_name] = label_encoder
                
                
    
    # Create a list of tuples containing column name and null percentage
    null_percentages = [(column_name, calculate_null_percentage(df[column_name])) for column_name in df.columns]

    # Sort the list based on null percentages in descending order
    null_percentages.sort(key=lambda x: x[1], reverse=True)

    # Process columns starting from the one with the highest null percentage
    for column_name, null_percentage in null_percentages:
        column = df[column_name]

        if null_percentage <= 30:
            print(f"Dropping rows with null values in '{column_name}'... because {null_percentage}")
            df = df[~column.isnull()]
        else:
            fill_nulls(column)

        numeric_nulls, string_nulls = split_nulls(column)
        print(f"Numeric Nulls in '{column_name}':\n{numeric_nulls} because {null_percentage}")
        print(f"String Nulls in '{column_name}':\n{string_nulls} because {null_percentage}")
    

    # Used to know which one is NUmbers in type Object like "10" or Which one is object ro make label encoding or which one numerical column 
    # Comming Soon ==>> Work with Date Format (It has some erros now)
    
                
                
    return df, label_encoders


# In[ ]:


layout = [
    [sg.Text("Select a file:")],
    [sg.InputText(key="file_path"), sg.FileBrowse(file_types=(("CSV files", "*.csv"), ("Excel files", "*.xlsx")))],
    [sg.Button("OK"), sg.Button("Cancel")]
]

window = sg.Window("File Selection", layout, finalize=True)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Cancel":
        break
    elif event == "OK":
        file_path = values["file_path"]
        print("Selected file:", file_path)
        break

window.close()
        
processed_df, label_encoders = preprocess_data(file_path)

print("Processed Data Types:")
print(processed_df.dtypes)
print("#" * 100)

print("Label Encoders:")
print(label_encoders)


# In[ ]:


column_names = list(processed_df.columns)
column_dic = {index: column_name for index, column_name in enumerate(column_names)}

layout = [
    [sg.Text("Select columns to drop:")],
    *[
        [sg.Checkbox(column_name, key=f"checkbox_{index}")] for index, column_name in column_dic.items()
    ],
    [sg.Button("OK"), sg.Button("Cancel")]
]

window = sg.Window("Column Selection", layout, finalize=True)

columns_to_drop = []

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Cancel":
        break

    if event == "OK":
        for index, column_name in column_dic.items():
            if values[f"checkbox_{index}"]:
                columns_to_drop.append(column_name)
        break

window.close()

if columns_to_drop:
    processed_df.drop(columns=columns_to_drop, axis=1, inplace=True)
    label_encoders = {key: value for key, value in label_encoders.items() if key not in columns_to_drop}
    print("Columns dropped successfully:", columns_to_drop)
else:
    print("No columns were dropped.")


# In[ ]:


column_names = list(processed_df.columns)

layout = [
    [sg.Text("Select your target column:")],
    *[
        [sg.Radio(column_name, "target_column", key=f"radio_{index}")] for index, column_name in enumerate(column_names)
    ],
    [sg.Button("OK")]
]

window = sg.Window("Target Column Selection", layout, finalize=True)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "OK":
        target_index = None
        for index, column_name in enumerate(column_names):
            if values[f"radio_{index}"]:
                target_index = index
                break

        if target_index is not None:
            target_column = column_names[target_index]
            print(f"Target column: {target_column}")
        else:
            print("No target column selected.")
        break

window.close()


# In[ ]:


sns.set(style="whitegrid")
inverse_label_encoders = list(label_encoders.keys())
for i in range(len(inverse_label_encoders)):
    processed_df[inverse_label_encoders[i]] = label_encoders[inverse_label_encoders[i]].inverse_transform(processed_df[inverse_label_encoders[i]])

print(processed_df.head())
all_columns = processed_df.columns.tolist()
all_columns.remove(target_column)

for column in all_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=processed_df, x=column, bins=20, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    
inverse_label_encoders = list(label_encoders.keys())
for i in range(len(inverse_label_encoders)):
    processed_df[inverse_label_encoders[i]] = label_encoders[inverse_label_encoders[i]].fit_transform(processed_df[inverse_label_encoders[i]])
    
print(processed_df.head())


# In[ ]:


X = processed_df.drop(target_column, axis=1)
y = processed_df[target_column]
problem_layout = [
    [sg.Text("Choose the problem type:")],
    [sg.Radio("Classification", "problem_type", key="classification"), sg.Radio("Regression", "problem_type", key="regression")],
    [sg.Button("Next")]
]

algorithm_layout = [
    [sg.Text("Available algorithms:")],
    [sg.Listbox(values=[], size=(30, 6), key="algorithm_list")],
    [sg.Button("Run")]
]

layout = [
    [sg.Column(problem_layout, key="problem_column"), sg.Column(algorithm_layout, visible=False, key="algorithm_column")]
]

window = sg.Window("Algorithm Selection", layout, finalize=True)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "Next":
        problem_type = "classification" if values["classification"] else "regression"

        if problem_type == "classification":
            algorithms = ["Logistic Regression", "K-Nearest Neighbors", "XGBoost", "Support Vector Machine"]
            scoring_metric = "accuracy"
        else:
            algorithms = ["Linear Regression", "Lasso", "Ridge", "XGBoost", "K-Nearest Neighbors", "Support Vector Machine"]
            scoring_metric = "r2_score"

        window["algorithm_list"].update(values=algorithms)
        window["problem_column"].update(visible=False)
        window["algorithm_column"].update(visible=True)

    if event == "Run":
        chosen_algorithm = values["algorithm_list"][0]

        if problem_type == "classification":
            if chosen_algorithm == "Logistic Regression":
                model = LogisticRegression()
            elif chosen_algorithm == "XGBoost":
                model = XGBClassifier()
            elif chosen_algorithm == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            elif chosen_algorithm == "Support Vector Machine":
                model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            else:
                sg.popup("Invalid algorithm choice. Exiting.")
                break
        else:
            if chosen_algorithm == "Linear Regression":
                model = LinearRegression()
            elif chosen_algorithm == "Lasso":
                model = Lasso()
            elif chosen_algorithm == "Ridge":
                model = Ridge()
            elif chosen_algorithm == "XGBoost":
                model = XGBRegressor()
            elif chosen_algorithm == "K-Nearest Neighbors":
                model = KNeighborsRegressor()
            elif chosen_algorithm == "Support Vector Machine":
                model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
            else:
                sg.popup("Invalid algorithm choice. Exiting.")
                break

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred,average="macro")
            recall = recall_score(y_test, y_pred,average="macro")
            f1score = f1_score(y_test, y_pred,average="macro")
            sg.popup(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1Score: {f1score:.2f}")
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            sg.popup(f"R2 Score: {r2:.2f}\nMean Absolute Error: {mae:.2f}\nRoot Mean Squared Error: {rmse:.2f}\nMean Squared Error: {mse:.2f}")
        break

window.close()


# In[ ]:




