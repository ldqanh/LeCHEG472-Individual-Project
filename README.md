# Fake Job Posting Detector

A sophisticated Streamlit application that leverages machine learning to detect fraudulent job postings. The app employs multiple ML models, comprehensive data preprocessing, and interactive visualizations to provide accurate fraud detection and insights. The application can be found at https://lecheg472-individual-project-kshfsbmubj4dgnjwkn6xca.streamlit.app/  

## Features

### Data Processing
- **US Job Filtering**: Focuses on US job market data
- **Text Cleaning**: 
  - Removes URLs, HTML tags, and special characters
  - Normalizes text formatting
  - Handles missing values
- **Feature Engineering**:
  - TF-IDF vectorization with n-gram support
  - Text length analysis
  - Outlier detection and removal

### Machine Learning Models
- **Multiple Classifiers**:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Class Imbalance Handling**:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Balanced class weights

### Visualization & Analysis
- **Interactive Charts**:
  - Text length distributions
  - Class distribution analysis
  - Model performance comparisons
- **Statistical Insights**:
  - Missing value analysis
  - Summary statistics
  - Model performance metrics

### Real-time Prediction
- Interactive job description testing
- Confidence scores for predictions
- Detailed analysis of prediction factors

## Prerequisites

### Software Requirements
- Python 3.8 or higher
- Git (for cloning repository)

### Required Python Packages
```bash
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/fake-job-detector.git
cd fake-job-detector
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the application:
- Open your web browser
- Navigate to http://localhost:8501

3. Using the App:
   - Upload a CSV file containing job posting data
   - View data preprocessing insights and visualizations
   - Train and compare multiple ML models
   - Test individual job descriptions for fraud detection

## Data Format

The input CSV should contain the following columns:
- title
- location
- company_profile
- description
- requirements
- benefits
- fraudulent (0 for legitimate, 1 for fraudulent)

## Model Performance

Current model performance metrics (as of latest testing):
- SVM: 98.27% accuracy
- XGBoost: 98.10% accuracy
- Logistic Regression: 97.87% accuracy
- Random Forest: 97.62% accuracy

All models show strong performance in detecting both legitimate and fraudulent job postings, with particular emphasis on minimizing false positives.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- Dataset: Employment Scam Aegean Dataset (EMSCAD)
- NLTK Project for text processing capabilities
- Scikit-learn community for machine learning tools
- Streamlit team for the web application framework


## Future Improvements

- Add deep learning models
- Implement feature importance analysis
- Add multi-language support
- Enhance visualization capabilities
- Add model explainability features
