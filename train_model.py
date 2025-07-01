import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os

# ✅ Check file exists
file_path = "dataset/Reviews.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"'{file_path}' not found. Please upload the dataset to the 'dataset/' folder.")

# ✅ Load and select required columns
df = pd.read_csv(file_path)

# ✅ Print columns to debug (optional)
print("Columns in CSV:", df.columns.tolist())

# ✅ Select and clean necessary columns
df = df[['Text', 'Score']].dropna()

# ✅ Label reviews: 1 = Fake (Score ≤ 3), 0 = Genuine (Score ≥ 4)
df['label'] = df['Score'].apply(lambda x: 1 if x <= 3 else 0)

# ✅ Optional: Sample 5000 for fast training/testing
df = df.sample(n=5000, random_state=42)

# ✅ Create ML pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=200))
])

# ✅ Train model
pipeline.fit(df['Text'], df['label'])

# ✅ Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved to model.pkl")
