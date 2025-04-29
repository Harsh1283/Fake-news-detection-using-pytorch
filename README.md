This project is a **Fake News Detection System** built using **PyTorch**. It uses a feed-forward neural network with a Bag-of-Words (BoW) vectorizer to classify news headlines as either **REAL** or **FAKE**.



 🚀 Features

- Trained using a simple 3-layer fully connected neural network.
-Uses TF-IDF vectorization on headlines.
-Web app built with Streamlit for real-time prediction.
-Detects whether a given news headline is likely Real or Fake.


 🧠 Model Architecture

- Input Layer: 5000-dimensional BoW vector
- Hidden Layers: [512, 256] neurons with ReLU and Dropout
- Output Layer: 1 neuron with Sigmoid
- Loss: Binary Cross Entropy
- Optimizer: Adam


🧾 Files
-fakenews_model.pth: Trained PyTorch model
-vectorizer.pkl: TF-IDF vectorizer used during training
-app.py: Streamlit app code
-tempCodeRunnerFile.py: Model training script
-confusion_matrix.png, training_curves.png: Evaluation plots


## ⚙️ Installation

1. Clone the repository:

bash--
git clone https://github.com/yourusername/fake-news-detection-pytorch.git
cd fake-news-detection-pytorch


Install dependencies:
pip install torch scikit-learn matplotlib

Testing the Model
To test the model with your own headlines:
python app.py

 first install all the requirements 
--pip install -r requirements.txt

Run the sreamlit app
--streamlit run app1.py


