import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

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

def train_neural_network(X_train, y_train):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                 patience=3,
                                 restore_best_weights=True)
    
    model.fit(X_train.toarray(), y_train,
              epochs=20,
              batch_size=32,
              validation_split=0.2,
              callbacks=[early_stopping],
              verbose=0)
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    if isinstance(model, Sequential):
        y_pred = (model.predict(X_test.toarray()) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, conf_matrix

def plot_confusion_matrices(conf_matrices, model_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (conf_matrix, model_name) in enumerate(zip(conf_matrices, model_names)):
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[i])
        axes[i].set_title(f'{model_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def compare_models(df):
    # Prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)
    
    # Train models
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    
    print("Training Neural Network...")
    nn_model = train_neural_network(X_train, y_train)
    
    # Evaluate models
    models = [rf_model, xgb_model, nn_model]
    model_names = ['Random Forest', 'XGBoost', 'Neural Network']
    accuracies = []
    conf_matrices = []
    
    for model, name in zip(models, model_names):
        accuracy, conf_matrix = evaluate_model(model, X_test, y_test, name)
        accuracies.append(accuracy)
        conf_matrices.append(conf_matrix)
    
    # Plot confusion matrices
    conf_matrix_fig = plot_confusion_matrices(conf_matrices, model_names)
    
    # Find best model
    best_model_index = np.argmax(accuracies)
    best_model = models[best_model_index]
    best_model_name = model_names[best_model_index]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {accuracies[best_model_index]:.4f}")
    
    return best_model, vectorizer, conf_matrix_fig

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
    
    st.header("Model Comparison")
    
    if st.button("Compare Models"):
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
            if isinstance(st.session_state['best_model'], Sequential):
                prediction = (st.session_state['best_model'].predict(input_vector.toarray()) > 0.5).astype(int)
            else:
                prediction = st.session_state['best_model'].predict(input_vector)
            
            if prediction[0] == 1:
                st.warning("This job posting appears to be fraudulent")
            else:
                st.success("This job posting appears to be legitimate")

else:
    st.write("Please upload a CSV file to begin.")
