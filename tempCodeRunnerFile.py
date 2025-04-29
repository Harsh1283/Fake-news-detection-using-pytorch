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