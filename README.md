# Cyberbullying Detection System

A comprehensive machine learning system for detecting cyberbullying content in text using Streamlit and scikit-learn.

## Features

- **Real-time Predictions**: Analyze text and get instant cyberbullying predictions with confidence scores
- **Interactive Dashboard**: Multi-page Streamlit interface with analytics and visualizations
- **Model Training Pipeline**: Complete ML pipeline with data preprocessing, SMOTE resampling, and model evaluation
- **Performance Analytics**: View detailed metrics including accuracy, precision, recall, F1-score, and confusion matrices
- **Feature Explanation**: Understand which words contribute most to bullying/non-bullying predictions
- **File Management**: Upload custom models, vectorizers, and datasets at runtime

## Project Structure

```
├── app.py                                    # Main Streamlit application
├── cyberbullying_detection_system.ipynb      # ML training pipeline and EDA
├── save_test_split.py                        # Script to generate test split files
├── aggression_parsed_dataset.csv             # Training dataset
├── cyberbullying_model_lr.pkl                # Trained Logistic Regression model
├── tfidf_vectorizer.pkl                      # Fitted TF-IDF vectorizer
├── X_test_sparse.npz                         # Test set features
├── y_test.npy                                # Test set labels
└── README.md                                 # This file
```

## Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/doyancha/Cyberbullying_Detection_System.git
cd Cyberbullying_Detection_System
```

2. **Create a virtual environment** (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK resources**:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` with the following pages:

- **🏠 Live Detection**: Analyze single texts for cyberbullying
- **📊 Analytics Dashboard**: View model performance metrics and visualizations
- **⚙️ System Info**: Check model configuration and training details
- **📚 About**: Project information and ethical considerations

### Train the Model

To retrain the model with new data, use the Jupyter notebook:

```bash
jupyter notebook cyberbullying_detection_system.ipynb
```

The notebook includes:
- Data loading and exploration
- Text preprocessing
- SMOTE resampling
- TF-IDF vectorization
- Model training with GridSearchCV
- Evaluation metrics and visualizations
- Model persistence

### Generate Test Split Files

If test split files are missing, regenerate them:

```bash
python save_test_split.py
```

This creates:
- `X_test_sparse.npz` - Test features
- `y_test.npy` - Test labels

## Model Details

### Architecture
- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF (5,000 features, unigrams + bigrams)
- **Class Balance**: SMOTE oversampling
- **Train/Test Split**: 80/20 with stratification

### Performance
- **Accuracy**: ~81%
- **Precision**: ~96.4%
- **Recall**: ~80.2%
- **F1-Score**: ~87.6%

## Data Preprocessing

Text cleaning includes:
- Lowercase conversion
- URL and mention removal
- Special character removal
- Stopword filtering
- (Optional) Lemmatization via NLTK

## Ethical Considerations

- This system is designed to detect harmful content
- Should be used as a support tool, not a final decision-maker
- Regular audits recommended to identify and mitigate bias
- Transparent communication essential when deploying

## License

This project is open source and available under the MIT License.

## Author

Mir Shahadut Hossain

## References

- Dataset: https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset?utm_source=chatgpt.com&select=aggression_parsed_dataset.csv
- Scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/
