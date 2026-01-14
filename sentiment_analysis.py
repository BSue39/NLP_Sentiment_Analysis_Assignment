import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r'C:\Users\bfens\OneDrive\Documents\NLP_Sentiment_Analysis_Assignment\exampledataset2.csv')

# Select Columns
text_column = "text"
sentiment_column = "sentiment"

# Split into features (X) and target (y)
X = df[text_column]
y = df[sentiment_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Vectorize text data
vectorizer = CountVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Intialize SVM classifier
model = SVC()

# Train the model
model.fit(X_train_vectors, y_train)

# Make predictions
y_pred = model.predict(X_test_vectors)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Model Accuracy:", accuracy)