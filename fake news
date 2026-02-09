import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv("news.csv")  
# columns: text, label

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# 3. Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train the AI model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. Test accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# 6. Test with custom news
def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "REAL NEWS" if prediction[0] == 1 else "FAKE NEWS"

# Example
sample_news = "Government announces new education policy for 2026"
print(predict_news(sample_news))
