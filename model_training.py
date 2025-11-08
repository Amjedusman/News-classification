import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# -------------------------
# Load AG News dataset
# -------------------------
train_df = pd.read_csv("train.csv")  # Columns: Class Index, Title, Description
test_df = pd.read_csv("test.csv")

# Map numeric labels to text categories
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

train_df["category"] = train_df["Class Index"].map(label_map)
test_df["category"] = test_df["Class Index"].map(label_map)

# Use the 'Title' column as the headline text
X = train_df["Title"]
y = train_df["category"]

# -------------------------
# Split Data
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# -------------------------
# Train Model
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -------------------------
# Evaluate Model
# -------------------------
y_pred = model.predict(X_val_tfidf)
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# -------------------------
# Save Model & Vectorizer
# -------------------------
with open("news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")
