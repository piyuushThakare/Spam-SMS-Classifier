import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #get base directory
DATA_PATH = os.path.join(BASE_DIR, "data_set", "spam.csv") #path to dataset

data = pd.read_csv(DATA_PATH, encoding="latin-1") #load dataset with latin-1 encoding to handle special characters

# 2. Clean dataset
data = data[['v1', 'v2']] #keep only relevant columns
data.columns = ['label', 'text'] #rename columns for clarity

# 3. Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1}) #ham=0, spam=1

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split( # split the data into training and testing sets
    data['text'], #features
    data['label'], #labels
    test_size=0.2, # 20% for testing
    random_state=42 # reproducibility used in app.py reproducibility means same results every time
)

# 5. Text vectorization
vectorizer = TfidfVectorizer(stop_words='english') #removes common words like "the", "is", etc.
X_train_vec = vectorizer.fit_transform(X_train) #fit_transform learns the vocabulary and idf from training data and transforms training data into vectors
X_test_vec = vectorizer.transform(X_test) #transform test data into vectors using the learned vocabulary and idf

# 6. Train model
model = MultinomialNB() #Naive Bayes classifier suitable for discrete features like word counts
model.fit(X_train_vec, y_train) #fit the model with training data

# 7. Evaluate
y_pred = model.predict(X_test_vec) #predict labels for test data
accuracy = accuracy_score(y_test, y_pred) #calculate accuracy
print(f"Model Accuracy: {accuracy * 100:.2f}%") #print accuracy percentage

# 8. Save model & vectorizer
MODEL_DIR = os.path.join(BASE_DIR, "model") #directory to save model and vectorizer this code creates the directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True) #create directory if it doesn't exist

with open(os.path.join(MODEL_DIR, "spam_model.pkl"), "wb") as f: #save the trained model
    pickle.dump(model, f) #dump the model object into the file

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f: #save the vectorizer
    pickle.dump(vectorizer, f) #dump the vectorizer object into the file

print("✅ Model and vectorizer saved successfully") #confirmation message



