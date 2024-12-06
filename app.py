pip install streamlit pandas numpy seaborn scikit-learn 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import nltk


# Initialize nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))


# Set the title
st.title("Fake Job Post Detection")

# Sidebar for user input
st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    
    # Preprocess data
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df[text_columns] = df[text_columns].fillna(' ')
    df['location'].fillna('Unknown', inplace=True)
    df['department'].fillna('Unknown', inplace=True)
    df['salary_range'].fillna('Not Specified', inplace=True)
    df['employment_type'].fillna('Not Specified', inplace=True)
    df['required_experience'].fillna('Not Specified', inplace=True)
    df['required_education'].fillna('Not Specified', inplace=True)
    df['industry'].fillna('Not Specified', inplace=True)
    df['function'].fillna('Not Specified', inplace=True)

    # Combine text columns
    if 'text' not in df.columns:
        df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + \
                     ' ' + df['requirements'] + ' ' + df['benefits']
        df['text'] = df['text'].apply(lambda x: x.lower())
        df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    st.write("Data Preview:")
    st.dataframe(df.head())

    # Data visualization
    st.write("Category Distribution:")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='fraudulent', ax=ax)
    st.pyplot(fig)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.fraudulent, test_size=0.3, random_state=42)

    # Text Vectorization
    vect = CountVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Model Training
    dt = DecisionTreeClassifier()
    dt.fit(X_train_dtm, y_train)

    # Evaluate the model
    y_pred_class = dt.predict(X_test_dtm)
    st.write("Classification Accuracy:", accuracy_score(y_test, y_pred_class))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred_class))
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_class)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

    # Job prediction
    st.write("Job Prediction:")
    input_text = st.text_input("Enter job description:")
    if input_text:
        input_data_features = vect.transform([input_text])
        prediction = dt.predict(input_data_features)
        if prediction[0] == 1:
            st.warning("Fraudulent Job")
        else:
            st.success("Real Job")

else:
    st.write("Please upload a CSV file to begin.")
