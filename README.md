# Fake Job Posting Detector

This Streamlit app analyzes job postings to detect fraudulent job offers using machine learning. It provides data insights, model training and evaluation, and allows users to test job descriptions for potential fraudulence.

## Features

- **Data Upload**: Upload a CSV file of job postings for analysis.
- **Visualization**: Explore data distribution and trends through interactive charts.
- **Model Training**: Automatically preprocesses data and trains a decision tree classifier.
- **Prediction**: Test job descriptions to check if they are fraudulent.

## Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fake-job-detector.git
   cd fake-job-detector

2. Run the app
3. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
Open the provided URL in your browser (e.g., http://localhost:8501).

## Notes
Ensure your Python environment's SSL certificates are updated to avoid NLTK download issues.
