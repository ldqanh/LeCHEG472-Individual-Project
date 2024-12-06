import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
import nltk
import re
import missingno as msno
from scipy import stats

# Initialize nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def analyze_word_frequencies(df):
    """Analyze and visualize word frequencies in fraudulent and non-fraudulent postings"""
    st.subheader("Word Frequency Analysis")

    # Separate fraudulent and non-fraudulent job postings
    fraudulent_jobs = df[df['fraudulent'] == 1]['text']
    non_fraudulent_jobs = df[df['fraudulent'] == 0]['text']

    def get_word_freq(text_series):
        words = ' '.join(text_series).split()
        return pd.Series(words).value_counts()

    # Plot top words for fraudulent jobs
    fraud_word_freq = get_word_freq(fraudulent_jobs).head(20)
    fig_fraud = px.bar(fraud_word_freq, 
                      x=fraud_word_freq.index, 
                      y=fraud_word_freq.values,
                      title='Top Words in Fraudulent Job Postings',
                      labels={'index': 'Words', 'y': 'Frequency'},
                      color=fraud_word_freq.values,
                      color_continuous_scale='Reds')
    st.plotly_chart(fig_fraud)

    # Plot top words for non-fraudulent jobs
    non_fraud_word_freq = get_word_freq(non_fraudulent_jobs).head(20)
    fig_non_fraud = px.bar(non_fraud_word_freq,
                          x=non_fraud_word_freq.index,
                          y=non_fraud_word_freq.values,
                          title='Top Words in Non-Fraudulent Job Postings',
                          labels={'index': 'Words', 'y': 'Frequency'},
                          color=non_fraud_word_freq.values,
                          color_continuous_scale='Blues')
    st.plotly_chart(fig_non_fraud)

    # Compare word frequencies
    st.write("\nUnique words in fraudulent vs non-fraudulent postings:")
    fraud_unique = set(fraud_word_freq.index) - set(non_fraud_word_freq.index)
    non_fraud_unique = set(non_fraud_word_freq.index) - set(fraud_word_freq.index)
    st.write("Words unique to fraudulent postings:", list(fraud_unique))
    st.write("Words unique to non-fraudulent postings:", list(non_fraud_unique))

def plot_class_distribution(df):
    """Plot the distribution of fraudulent vs non-fraudulent postings"""
    st.subheader("Class Distribution")

    fig = px.histogram(df, 
                      x='fraudulent',
                      title='Distribution of Fraudulent vs Non-Fraudulent Job Postings',
                      labels={'fraudulent': 'Fraudulent'},
                      color='fraudulent',
                      color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    
    fig.update_layout(
        xaxis_title='Fraudulent',
        yaxis_title='Count',
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=14),
        xaxis=dict(gridcolor='gray'),
        yaxis=dict(gridcolor='gray')
    )
    
    st.plotly_chart(fig)

    # Add statistics
    fraud_percent = (df['fraudulent'].mean() * 100)
    st.write(f"Percentage of fraudulent postings: {fraud_percent:.2f}%")
    st.write(f"Total number of postings: {len(df)}")
    st.write(f"Number of fraudulent postings: {df['fraudulent'].sum()}")
    st.write(f"Number of legitimate postings: {len(df) - df['fraudulent'].sum()}")

def handle_missing_values(df):
    """Handle missing values in all columns"""
    st.subheader("Missing Value Handling")

    # Store initial missing value counts
    initial_missing = df.isnull().sum()
    
    # Text columns
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df[text_columns] = df[text_columns].fillna(' ')
    
    # Categorical columns
    categorical_columns = ['location', 'department', 'salary_range', 'employment_type',
                         'required_experience', 'required_education', 'industry', 'function']
    for col in categorical_columns:
        if col in df.columns:
            df[col].fillna('Not Specified', inplace=True)
    
    # Display missing value handling results
    final_missing = df.isnull().sum()
    
    # Create comparison dataframe
    missing_comparison = pd.DataFrame({
        'Initial Missing': initial_missing,
        'After Handling': final_missing,
        'Difference': initial_missing - final_missing
    })
    
    st.write("Missing Values Before and After Handling:")
    st.dataframe(missing_comparison[missing_comparison['Initial Missing'] > 0])

def preprocess_data(df):
    """Main preprocessing function"""
    st.write("Starting data preprocessing and analysis...")
    
    # Handle missing values
    handle_missing_values(df)
    
    # Plot class distribution
    plot_class_distribution(df)
    
    # Analyze word frequencies
    df['text'] = df[['title', 'company_profile', 'description', 'requirements', 'benefits']].apply(
        lambda x: ' '.join(x.dropna()), axis=1)
    analyze_word_frequencies(df)
    
    # Clean text
    for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
        df[col] = df[col].apply(clean_text)
    
    # Filter for US jobs only
    df = df[df['location'].str.contains('US', na=False)]
    st.write("\nDataset Shape after filtering US jobs:", df.shape)
    
    # Create numerical features and handle outliers
    numerical_features = create_numerical_features(df)
    
    # Analyze correlations
    analyze_correlations(df, numerical_features)
    
    # Analyze outliers
    analyze_outliers(numerical_features)
    
    # Generate summary statistics
    generate_summary_stats(df, numerical_features)
    
    return df
def prepare_data(df):
    # Text preprocessing
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df['text'] = df[text_columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['fraudulent']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test, vectorizer

def display_data_analysis(df):
    st.header("Data Analysis Dashboard")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview & Distribution", 
        "Missing Data", 
        "Word Analysis",
        "Correlations",
        "Model Results"
    ])
    
    with tab1:
        # Distribution of fraudulent vs non-fraudulent
        fig = px.histogram(df, 
                          x='fraudulent', 
                          title='Distribution of Fraudulent vs Non-Fraudulent Job Postings',
                          labels={'fraudulent': 'Fraudulent'},
                          color='fraudulent',
                          color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        
        fig.update_layout(
            xaxis_title='Fraudulent',
            yaxis_title='Count',
            title_x=0.5
        )
        st.plotly_chart(fig)
        
        # Dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Total Samples:", len(df))
            st.write("Legitimate Jobs:", len(df[df['fraudulent'] == 0]))
        with col2:
            st.write("Fraudulent Jobs:", len(df[df['fraudulent'] == 1]))
            st.write("Fraud Percentage:", f"{(df['fraudulent'].mean() * 100):.2f}%")
    
    with tab2:
        st.subheader("Missing Values Analysis")
        # Missing values heatmap
        missing_data = df.isnull().sum()
        fig = px.bar(x=missing_data.index, 
                    y=missing_data.values,
                    title="Missing Values by Column")
        st.plotly_chart(fig)
        
        # Missing values table
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Values': missing_data.values,
            'Percentage': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df)
    
    with tab3:
        st.subheader("Word Analysis")
        
        # Separate fraudulent and non-fraudulent job postings
        fraudulent_jobs = df[df['fraudulent'] == 1]['text']
        non_fraudulent_jobs = df[df['fraudulent'] == 0]['text']
        
        def plot_top_words(text_series, title):
            word_freq = pd.Series(' '.join(text_series).split()).value_counts().head(20)
            fig = px.bar(word_freq, 
                        x=word_freq.index, 
                        y=word_freq.values, 
                        title=title,
                        labels={'index': 'Words', 'y': 'Frequency'})
            st.plotly_chart(fig)
        
        plot_top_words(fraudulent_jobs, 'Top Words in Fraudulent Job Postings')
        plot_top_words(non_fraudulent_jobs, 'Top Words in Non-Fraudulent Job Postings')
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        # Create numerical features for correlation
        numerical_features = pd.DataFrame()
        text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        
        for col in text_columns:
            numerical_features[f'{col}_length'] = df[col].str.len()
            numerical_features[f'{col}_word_count'] = df[col].str.split().str.len()
        
        numerical_features['fraudulent'] = df['fraudulent']
        
        # Plot correlation matrix
        corr = numerical_features.corr()
        fig = px.imshow(corr,
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig)
    
    with tab5:
        st.subheader("Model Results")
        # This tab will be populated after model training

def display_model_results(accuracies, reports, conf_matrices):
    """Display model results in the Model Results tab"""
    st.subheader("Model Performance Comparison")
    
    # Create accuracy comparison
    accuracy_df = pd.DataFrame({
        'Model': list(accuracies.keys()),
        'Accuracy': list(accuracies.values())
    })
    
    fig = px.bar(accuracy_df, 
                 x='Model', 
                 y='Accuracy',
                 title='Model Accuracy Comparison')
    st.plotly_chart(fig)
    
    # Display classification reports
    for model_name, report in reports.items():
        st.write(f"\n{model_name} Classification Report:")
        st.text(report)

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, 
                                    max_depth=10,
                                    min_samples_split=5,
                                    class_weight='balanced',
                                    random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    xgb_model = XGBClassifier(n_estimators=100,
                             max_depth=5,
                             learning_rate=0.1,
                             scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                             random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(class_weight='balanced',
                                 max_iter=1000,
                                 random_state=42)
    lr_model.fit(X_train, y_train)
    return lr_model

def train_svm(X_train, y_train):
    svm_model = LinearSVC(class_weight='balanced',
                         max_iter=2000,
                         random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    st.write(f"\n{model_name} Results:")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write("\nClassification Report:")
    st.text(report)
    
    return accuracy, report, conf_matrix
    
def plot_confusion_matrices(conf_matrices, model_names):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, (conf_matrix, model_name) in enumerate(zip(conf_matrices, model_names)):
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{model_name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def compare_models(df):
    # Prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)
    
    # First, show data analysis
    display_data_analysis(df)
    
    # Initialize lists to store results
    models = []
    model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM']
    accuracies = {}
    reports = {}
    conf_matrices = []
    
    # Train and evaluate Random Forest
    st.write("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    models.append(rf_model)
    accuracy, report, conf_matrix = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    accuracies["Random Forest"] = accuracy
    reports["Random Forest"] = report
    conf_matrices.append(conf_matrix)
    
    # Train and evaluate XGBoost
    st.write("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    models.append(xgb_model)
    accuracy, report, conf_matrix = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    accuracies["XGBoost"] = accuracy
    reports["XGBoost"] = report
    conf_matrices.append(conf_matrix)
    
    # Train and evaluate Logistic Regression
    st.write("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    models.append(lr_model)
    accuracy, report, conf_matrix = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    accuracies["Logistic Regression"] = accuracy
    reports["Logistic Regression"] = report
    conf_matrices.append(conf_matrix)
    
    # Train and evaluate SVM
    st.write("Training SVM...")
    svm_model = train_svm(X_train, y_train)
    models.append(svm_model)
    accuracy, report, conf_matrix = evaluate_model(svm_model, X_test, y_test, "SVM")
    accuracies["SVM"] = accuracy
    reports["SVM"] = report
    conf_matrices.append(conf_matrix)
    
    # Plot confusion matrices
    conf_matrix_fig = plot_confusion_matrices(conf_matrices, model_names)
    
    # Display model results
    display_model_results(accuracies, reports, conf_matrices)
    
    # Find best model
    best_model_name = max(accuracies.items(), key=lambda x: x[1])[0]
    best_model = models[model_names.index(best_model_name)]
    
    st.write(f"\nBest Model: {best_model_name}")
    st.write(f"Best Accuracy: {accuracies[best_model_name]:.4f}")
    
    return best_model, vectorizer, conf_matrix_fig


if st.button("Compare Models"):
    with st.spinner("Training and comparing models..."):
        best_model, vectorizer, conf_matrix_fig = compare_models(df)
        
        st.success("Model comparison completed!")
        st.pyplot(conf_matrix_fig)
        
        # Save best model and vectorizer for predictions
        st.session_state['best_model'] = best_model
        st.session_state['vectorizer'] = vectorizer
# Set the title
st.title("Fake Job Post Detection")

# Sidebar for user input
st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    
    # Data Preview
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Data Loading and Model Comparison section
st.header("Model Comparison")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="csv_uploader")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    
    # Show data preview
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Model comparison button
    if st.button("Compare Models", key="compare_models_button"):
        with st.spinner("Training and comparing models..."):
            best_model, vectorizer, conf_matrix_fig = compare_models(df)
            st.success("Model comparison completed!")
            st.pyplot(conf_matrix_fig)
            
            # Save best model and vectorizer for predictions
            st.session_state['best_model'] = best_model
            st.session_state['vectorizer'] = vectorizer
    
    # Prediction section
    if 'best_model' in st.session_state:
        st.header("Job Prediction with Best Model")
        input_text = st.text_area("Enter job description:")
        
        if input_text:
            # Transform input text
            input_vector = st.session_state['vectorizer'].transform([input_text])
            
            # Make prediction
            prediction = st.session_state['best_model'].predict(input_vector)
            
            if prediction[0] == 1:
                st.warning("This job posting appears to be fraudulent")
            else:
                st.success("This job posting appears to be legitimate")
else:
    st.write("Please upload a CSV file to begin.")
