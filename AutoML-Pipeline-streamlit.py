import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

st.title("Data Preprocessing and Machine Learning")

file_path = st.file_uploader("Select a file (CSV or Excel):", type=["csv", "xlsx"])
if file_path:
    st.write("Selected file:", file_path.name)

    def preprocess_data(file_path):
        label_encoders = {}

        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            st.error("Unsupported file format. Please select a CSV or Excel file.")
            return None, None

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
                if not converted_column.isna().all():
                    df[column_name] = converted_column
                else:
                    label_encoder = LabelEncoder()
                    encoded_values = label_encoder.fit_transform(df[column_name])
                    df[column_name] = encoded_values
                    label_encoders[column_name] = label_encoder

        null_percentages = [(column_name, calculate_null_percentage(df[column_name])) for column_name in df.columns]
        null_percentages.sort(key=lambda x: x[1], reverse=True)

        for column_name, null_percentage in null_percentages:
            column = df[column_name]

            if null_percentage <= 30:
                df = df[~column.isnull()]
            else:
                fill_nulls(column)

        return df, label_encoders

    processed_df, label_encoders = preprocess_data(file_path)

    if processed_df is not None:
        st.write("Processed Data Types:")
        st.write(processed_df.dtypes)

        st.write("Label Encoders:")
        st.write(label_encoders)

        columns_to_drop = st.multiselect("Select columns to drop:", processed_df.columns)
        if columns_to_drop:
            processed_df.drop(columns=columns_to_drop, axis=1, inplace=True)
            label_encoders = {key: value for key, value in label_encoders.items() if key not in columns_to_drop}
            st.write("Columns dropped successfully:", columns_to_drop)
        else:
            st.write("No columns were dropped.")

        target_column = st.selectbox("Select your target column:", processed_df.columns)
        if st.checkbox("Perform Data Preprocessing"):
                
            sns.set(style="whitegrid")

            inverse_label_encoders = list(label_encoders.keys())
            for i in range(len(inverse_label_encoders)):
                processed_df[inverse_label_encoders[i]] = label_encoders[inverse_label_encoders[i]].inverse_transform(processed_df[inverse_label_encoders[i]])

            st.write(processed_df.head())

            all_columns = processed_df.columns.tolist()
            all_columns.remove(target_column)

            for column in all_columns:
                plt.figure(figsize=(8, 6))
                sns.histplot(data=processed_df, x=column, bins=20, kde=True)
                plt.title(f"Distribution of {column}")
                plt.xlabel(column)
                plt.ylabel("Count")
                st.pyplot(plt)

            def Visualization(data):
                # Display summary statistics
                st.subheader("Summary Statistics:")
                st.write(data.describe())

                # Display data visualization
                st.subheader("Data Visualization:")

                # 1. Histogram for numerical columns
                numerical_cols = data.select_dtypes(include='number').columns
                for col in numerical_cols:
                    st.write(f"### {col} Histogram")
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data[col], kde=True)
                    st.pyplot()
                
                # 2. Box plots for numerical columns
                st.write("### Box Plots for Numerical Columns")
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=data[numerical_cols])
                plt.xticks(rotation=45)
                st.pyplot()

                # 3. Pair plot for numerical columns (scatter plots and histograms)
                st.write("### Pair Plot for Numerical Columns")
                numerical_subset = data[numerical_cols]
                sns.set(style="ticks")
                sns.pairplot(numerical_subset, diag_kind='kde')
                st.pyplot()

                # 4. Count plot for categorical columns
                categorical_cols = data.select_dtypes(exclude='number').columns
                for col in categorical_cols:
                    st.write(f"### {col} Count Plot")
                    plt.figure(figsize=(8, 6))
                    sns.countplot(data=data, x=col)
                    plt.xticks(rotation=45)
                    st.pyplot()

                # 5. Correlation heatmap for numerical columns
                st.write("### Correlation Heatmap")
                plt.figure(figsize=(10, 8))
                corr_matrix = numerical_subset.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                st.pyplot()

            Visualization(processed_df)

            inverse_label_encoders = list(label_encoders.keys())
            for i in range(len(inverse_label_encoders)):
                processed_df[inverse_label_encoders[i]] = label_encoders[inverse_label_encoders[i]].fit_transform(processed_df[inverse_label_encoders[i]])


        X = processed_df.drop(target_column, axis=1)
        y = processed_df[target_column]

        problem_type = st.radio("Choose the problem type:", ("Classification", "Regression"))
        if problem_type == "Classification":
            algorithms = ["Logistic Regression", "K-Nearest Neighbors", "XGBoost", "Support Vector Machine"]
            scoring_metric = "accuracy"
        else:
            algorithms = ["Linear Regression", "Lasso", "Ridge", "XGBoost", "K-Nearest Neighbors", "Support Vector Machine"]
            scoring_metric = "r2_score"

        chosen_algorithm = st.selectbox("Select an algorithm:", algorithms)

        if st.button("Run"):
            if problem_type == "Classification":
                if chosen_algorithm == "Logistic Regression":
                    model = LogisticRegression()
                elif chosen_algorithm == "XGBoost":
                    model = XGBClassifier()
                elif chosen_algorithm == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()
                elif chosen_algorithm == "Support Vector Machine":
                    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                else:
                    st.error("Invalid algorithm choice. Please select a valid algorithm.")
                    st.stop()
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
                    st.error("Invalid algorithm choice. Please select a valid algorithm.")
                    st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="macro")
                recall = recall_score(y_test, y_pred, average="macro")
                f1score = f1_score(y_test, y_pred, average="macro")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1score:.2f}")
            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                st.write(f"R2 Score: {r2:.2f}")
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
