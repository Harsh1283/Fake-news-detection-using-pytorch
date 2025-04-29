import torch
import pickle
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Define your model architecture
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = torch.sigmoid(self.layer3(x))
        return x

# Load or create vectorizer
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        print("✓ Loaded existing vectorizer")
except FileNotFoundError:
    print("Creating new vectorizer...")
    # Load datasets
    fake_df = pd.read_csv('fake.csv')
    true_df = pd.read_csv('true.csv')
    
    # Combine texts
    all_texts = pd.concat([fake_df['text'], true_df['text']], ignore_index=True)
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(all_texts)
    
    # Save the fitted vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Created and saved new vectorizer")

# Load model
input_dim = 5000  # same as during training
model = FakeNewsClassifier(input_dim)
model.load_state_dict(torch.load('fakenews_model.pt'))
model.eval()

# Prediction function
def predict_news(news_text):
    vectorized_text = vectorizer.transform([news_text]).toarray()
    input_tensor = torch.tensor(vectorized_text, dtype=torch.float32)
    output = model(input_tensor)
    predicted_class = torch.round(output).item()
    confidence = output.item()
    if predicted_class == 1.0:
        print(f"\n🟢 Prediction: REAL News ✅ (Confidence: {confidence:.2%})\n")
    else:
        print(f"\n🔴 Prediction: FAKE News ❌ (Confidence: {1-confidence:.2%})\n")

# Interactive Menu
print("📰 Fake News Detection System (Manual Testing)")
print("Type 'exit' to quit.\n")

while True:
    try:
        news_input = input("Enter a news headline to predict: ")
        if news_input.lower() == 'exit':
            print("Exiting... Goodbye! 👋")
            break
        predict_news(news_input)
    except KeyboardInterrupt:
        print("\nExiting... Goodbye! 👋")
        break
    except Exception as e:
        print(f"\nError: {str(e)}")
