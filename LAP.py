import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_squared_error, mean_absolute_error

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.title("Loan Approval Prediction")

# Display the process image when it is available.
if os.path.exists('process.jpg'):
    st.image('process.jpg', caption='Loan Approval Decision - The Journey Begins', use_column_width=True)

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Data preprocessing
    df.drop(columns='loan_id', inplace=True)
    df['Movable_assets'] = df[' bank_asset_value'] + df[' luxury_assets_value']
    df['Immovable_assets'] = df[' residential_assets_value'] + df[' commercial_assets_value']
    df.drop(columns=[' bank_asset_value', ' luxury_assets_value', ' residential_assets_value', ' commercial_assets_value'], inplace=True)

    # Label encoding
    df[' education'] = df[' education'].map({' Not Graduate': 0, ' Graduate': 1})
    df[' self_employed'] = df[' self_employed'].map({' No': 0, ' Yes': 1})
    df[' loan_status'] = df[' loan_status'].map({' Rejected': 0, ' Approved': 1})

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")

    # Number of Dependents
    fig, ax = plt.subplots()
    sns.countplot(x=' no_of_dependents', data=df, ax=ax).set_title('Number of Dependents')
    st.pyplot(fig)

    # Income vs Education
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(x=' education', y=' income_annum', data=df, ax=ax[0])
    sns.violinplot(x=' education', y=' income_annum', data=df, ax=ax[1])
    st.pyplot(fig)

    # Self-Employed vs Education
    fig, ax = plt.subplots()
    sns.countplot(x=' self_employed', data=df, hue=' education', ax=ax).set_title('Self Employed')
    st.pyplot(fig)

    # Loan Amount vs Loan Term
    fig, ax = plt.subplots()
    sns.lineplot(x=' loan_term', y=' loan_amount', data=df, ax=ax).set_title('Loan Amount vs. Loan Term')
    st.pyplot(fig)

    # CIBIL Score Distribution
    fig, ax = plt.subplots()
    sns.histplot(df[' cibil_score'], bins=30, kde=True, color='red', ax=ax)
    st.pyplot(fig)

    # Movable and Immovable Assets Distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(df['Movable_assets'], ax=ax[0], color='red')
    sns.histplot(df['Immovable_assets'], ax=ax[1], color='blue')
    st.pyplot(fig)

    # Loan Status by Number of Dependents
    fig, ax = plt.subplots()
    sns.countplot(x=' no_of_dependents', data=df, hue=' loan_status', ax=ax)
    st.pyplot(fig)

    # Loan Status by Education
    fig, ax = plt.subplots()
    sns.countplot(x=' education', hue=' loan_status', data=df, ax=ax).set_title('Loan Status by Education')
    st.pyplot(fig)

    # Income vs Loan Status
    fig, ax = plt.subplots()
    sns.violinplot(x=' loan_status', y=' income_annum', data=df, ax=ax)
    st.pyplot(fig)

    # Loan Amount vs Loan Term by Loan Status
    fig, ax = plt.subplots()
    sns.lineplot(x=' loan_term', y=' loan_amount', data=df, hue=' loan_status', ax=ax)
    st.pyplot(fig)

    # CIBIL Score vs Loan Status
    fig, ax = plt.subplots()
    sns.violinplot(x=' loan_status', y=' cibil_score', data=df, ax=ax)
    st.pyplot(fig)

    # Movable and Immovable Assets Distribution by Loan Status
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(x='Movable_assets', data=df, ax=ax[0], hue=' loan_status', multiple='stack')
    sns.histplot(x='Immovable_assets', data=df, ax=ax[1], hue=' loan_status', multiple='stack')
    st.pyplot(fig)

    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Scatterplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(x='Movable_assets', y=' loan_amount', data=df, ax=ax[0]).set_title('Movable_assets vs loan_amount')
    sns.scatterplot(x='Immovable_assets', y=' loan_amount', data=df, ax=ax[1]).set_title('Immovable_assets vs loan_amount')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(x=' income_annum', y=' loan_amount', data=df, ax=ax)
    st.pyplot(fig)

    # Split the data
    X = df.drop(' loan_status', axis=1)
    y = df[' loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree Classifier
    st.subheader("Decision Tree Classifier")
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    dtree_pred = dtree.predict(X_test)

    st.write("Training Accuracy:", dtree.score(X_train, y_train))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, dtree_pred), annot=True, ax=ax)
    st.pyplot(fig)
    st.write("Classification Report:")
    st.text(classification_report(y_test, dtree_pred))

    # Random Forest Classifier
    st.subheader("Random Forest Classifier")
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)

    st.write("Training Accuracy:", rfc.score(X_train, y_train))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True, ax=ax)
    st.pyplot(fig)
    st.write("Classification Report:")
    st.text(classification_report(y_test, rfc_pred))

    # Evaluation Metrics
    st.subheader("Evaluation Metrics")

    # Decision Tree Classifier Metrics
    st.write("Decision Tree Classifier:")
    dtree_r2_score = r2_score(y_test, dtree_pred)
    dtree_mse = mean_squared_error(y_test, dtree_pred)
    dtree_mae = mean_absolute_error(y_test, dtree_pred)
    st.write(f"R2 score: {dtree_r2_score:.4f}")
    st.write(f"Mean Squared Error: {dtree_mse:.4f}")
    st.write(f"Mean Absolute Error: {dtree_mae:.4f}")

    # Random Forest Classifier Metrics
    st.write("Random Forest Classifier:")
    rfc_r2_score = r2_score(y_test, rfc_pred)
    rfc_mse = mean_squared_error(y_test, rfc_pred)
    rfc_mae = mean_absolute_error(y_test, rfc_pred)

    st.write(f"R2 score: {rfc_r2_score:.4f}")
    st.write(f"Mean Squared Error: {rfc_mse:.4f}")
    st.write(f"Mean Absolute Error: {rfc_mae:.4f}")


# Display the approval image when it is available.
if os.path.exists('approved.jpg'):
    st.image('approved.jpg', caption='Application Approved: Sealing the Deal with a Stamp of Success', use_column_width=True)
