import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

# Load and combine datasets
print("Loading datasets...")
fake_df = pd.read_csv('fake.csv')
true_df = pd.read_csv('true.csv')

# Add labels to each dataset
fake_df['label'] = 0  # 0 for fake news
true_df['label'] = 1  # 1 for true news

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
print(f"Combined dataset shape: {df.shape}")

# Shuffle the combined dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare features and labels
X = df['text'].fillna('')
y = df['label'].values

# Vectorize text
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_vectorized.toarray())
y_tensor = torch.FloatTensor(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Initialize model
model = FakeNewsClassifier(X_vectorized.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Initialize lists to store metrics
train_losses = []
train_accuracies = []
val_accuracies = []

# Modified training loop with validation
epochs = 5
batch_size = 32
print("\nStarting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size].reshape(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (outputs >= 0.5).float()
        correct_train += (predictions == batch_y).sum().item()
        total_train += batch_y.size(0)

    # Calculate epoch metrics
    avg_loss = total_loss * batch_size / len(X_train)
    train_accuracy = correct_train / total_train
    
    # Validation accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        val_predictions = (test_outputs >= 0.5).float()
        val_accuracy = (val_predictions.flatten() == y_test).float().mean().item()
    
    # Store metrics
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Training Loss: {avg_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}\n")

# Plot training curves
plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# Confusion Matrix
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = (test_outputs >= 0.5).float()
    conf_matrix = confusion_matrix(y_test, predictions.flatten())
    
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions.flatten()))

# Save model and vectorizer
print("\nSaving model and vectorizer...")
torch.save(model.state_dict(), 'fakenews_model.pt')
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

def load_model_and_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    model = FakeNewsClassifier(5000)
    model.load_state_dict(torch.load('fakenews_model.pt'))
    model.eval()
    
    return model, vectorizer

# Function to predict new text
def predict_text(text, model, vectorizer):
    model.eval()
    with torch.no_grad():
        # Vectorize the input text
        text_vectorized = vectorizer.transform([text]).toarray()
        # Convert to tensor
        text_tensor = torch.FloatTensor(text_vectorized)
        # Get prediction
        prediction = model(text_tensor)
        return "REAL" if prediction.item() >= 0.5 else "FAKE"

# Test loading and prediction
loaded_model, loaded_vectorizer = load_model_and_vectorizer()
print("Model and vectorizer loaded successfully")

# Test prediction
test_text = "This is a sample news article"
prediction = predict_text(test_text, loaded_model, loaded_vectorizer)
print(f"\nSample prediction: {prediction}")
