import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from utils import preprocess_text

# Define directory for model files and static images
MODEL_DIR = 'model'
STATIC_IMG_DIR = 'static/images'

# Ensure necessary directories exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(STATIC_IMG_DIR):
    os.makedirs(STATIC_IMG_DIR)

def generate_and_save_charts(y_test, y_pred, metrics):
    """Generates and saves the confusion matrix and metrics bar chart
    into the STATIC_IMG_DIR for the Flask dashboard to display."""
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Fake (0)', 'True (1)'],
                yticklabels=['Fake (0)', 'True (1)'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    # SAVING TO STATIC DIRECTORY
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Chart saved: {STATIC_IMG_DIR}/confusion_matrix.png")

    # --- Metrics Bar Chart ---
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    
    # Add value labels on top of bars
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.ylim(0, 1.1)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    # SAVING TO STATIC DIRECTORY
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'metrics_bar_chart.png'))
    plt.close()
    
    print(f"Chart saved: {STATIC_IMG_DIR}/metrics_bar_chart.png")


def train_and_save_model(model_path=os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'), 
                         vectorizer_path=os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')):
    """
    Trains a Logistic Regression model on a fake news dataset and saves it with
    its corresponding TF-IDF vectorizer.
    """
    print("Starting model training...")
    try:
        # Load the dataset
        # NOTE: Assumes 'Fake.csv' and 'True.csv' are in a subdirectory named 'data/'
        fake_df = pd.read_csv('data/Fake.csv')
        true_df = pd.read_csv('data/True.csv')

        # Add a 'label' column to each dataframe
        fake_df['label'] = 0  # 0 for fake news
        true_df['label'] = 1   # 1 for real news

        # Combine the dataframes
        df = pd.concat([fake_df, true_df]).reset_index(drop=True)
        # --- DEBUG POINT 1 ---
        print(f"Debug: Dataset size after concatenation: {df.shape[0]} rows.")
    except FileNotFoundError:
        print("\nError: Dataset not found.")
        print("Please place 'Fake.csv' and 'True.csv' inside a folder named 'data/'.")
        return
        
    # --- Data Cleaning ---
    # 1. Drop rows where 'text' or 'label' might be missing
    # We prioritize the 'text' column, assuming the content is here.
    df = df.dropna(subset=['text', 'label'])
    
    # 2. Convert 'label' to integer type
    df['label'] = df['label'].astype(int)
    df = df.reset_index(drop=True)
    # --- DEBUG POINT 2 ---
    print(f"Debug: Dataset size after NaN removal: {df.shape[0]} rows.")

    # 3. Apply preprocessing
    print("Preprocessing text data...")
    df['text'] = df['text'].apply(preprocess_text)
    
    # Convert any resulting None values (if the utility function returns None) to an empty string.
    df['text'] = df['text'].fillna('') 

    # 4. Crucial Step: Remove rows where preprocessing resulted in an empty string
    df = df[df['text'].str.strip() != '']
    df = df.reset_index(drop=True)
    
    # --- DEBUG POINT 3 ---
    print(f"Debug: Dataset size after preprocessing and empty string removal: {df.shape[0]} rows.")

    if len(df) == 0:
        print("Error: Dataset is empty after cleaning. Cannot train model.")
        return

    # Split the dataset into training and testing sets
    X = df['text']
    y = df['label']
    
    # Ensure there is enough data for both classes after splitting
    if len(X) < 2:
        print("Error: Not enough samples left after cleaning to perform train/test split.")
        return
        
    # Using 20% test size (standard practice) and stratify for balanced classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Fitting TF-IDF Vectorizer...")
    # Initialize and fit the TF-IDF Vectorizer on the training data
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating model performance...")
    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    
    print("\n--- Model Evaluation Report ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Generate and save charts
    generate_and_save_charts(y_test, y_pred, metrics)

    print("\nSaving the model and vectorizer...")
    # Save the trained model and vectorizer to disk
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
        
    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)
        
    print("Model and vectorizer saved successfully!")
    print("Training complete.")

if __name__ == '__main__':
    # Run the training process when the script is executed
    train_and_save_model()
