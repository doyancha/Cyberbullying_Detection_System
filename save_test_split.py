"""
Quick script to save the test split from the trained model pipeline.
Run this after the notebook has trained the model.
"""
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import scipy.sparse as sp

# Download NLTK resources
nltk.download('stopwords', quiet=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('aggression_parsed_dataset.csv')

# Setup
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Simplified cleaning for speed - skip lemmatization"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Skip lemmatization for speed, just filter stopwords
    text = ' '.join([word for word in text.split() if word and word not in stop_words])
    return text.strip()

# Clean text
print("Cleaning text...")
df['cleaned_text'] = df['Text'].apply(clean_text)

# Vectorize
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['oh_label']

# Apply SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split (same as notebook)
print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Save test split
print("Saving test split...")
sp.save_npz('X_test_sparse.npz', X_test)
np.save('y_test.npy', y_test)

print(f"✓ Test split saved successfully!")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  y_test distribution: {np.bincount(y_test.astype(int))}")
