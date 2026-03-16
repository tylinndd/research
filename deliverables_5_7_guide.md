# Deliverables 5–7 Implementation Guide (Keras + Breast Cancer + SMS Spam)

This document summarizes **how to complete deliverables 5–7** and the **process** an agent should follow to implement them in code.

---

## Deliverable 5: Feed‑Forward Network (Binary Classification)

### Goal
Build a small feed‑forward neural network in **Keras** for **binary classification** using the **Breast Cancer** dataset, plot training/validation loss, assess under/overfitting, and report test metrics (accuracy, precision, recall, F1).

### Process
1. **Load dataset**
   - Use `sklearn.datasets.load_breast_cancer()`.
   - Create `X` (features) and `y` (labels).

2. **Split data**
   - Split into train/validation/test (e.g., 70/15/15) using `train_test_split` with a fixed `random_state`.
   - Example: split train+temp, then split temp into validation/test.

3. **Scale features**
   - Fit `StandardScaler` on **training** data only.
   - Transform train/val/test with the same scaler.

4. **Build Keras model**
   - Simple MLP: Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid).
   - Compile with:
     - Loss: `binary_crossentropy`
     - Optimizer: `adam`
     - Metrics: `accuracy`

5. **Train**
   - Use `model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=..., batch_size=...)`.
   - Store training history.

6. **Plot training/validation loss**
   - Use matplotlib to plot `history.history['loss']` and `history.history['val_loss']`.

7. **Assess under/overfitting**
   - If training loss decreases but validation loss rises → **overfitting**.
   - If both losses remain high/flat → **underfitting**.
   - Explain in code comments or output.

8. **Evaluate on test set**
   - Predict probabilities, threshold at 0.5 to get class labels.
   - Use `sklearn.metrics` to compute:
     - Accuracy
     - Precision
     - Recall
     - F1

### Outputs to include
- Loss plot (train vs validation)
- Fit assessment (over/underfitting)
- Test metrics (accuracy, precision, recall, F1)

---

## Deliverable 6: SMS Spam TF‑IDF + Logistic Regression + MLP

### Goal
Use the **SMS Spam** dataset to build TF‑IDF features, train Logistic Regression and a small MLP, compare metrics, and extract top 10 most informative words.

### Process
1. **Load dataset**
   - Use the SMS Spam dataset (CSV, typically with `label` and `text` columns).
   - Convert labels to binary (e.g., ham=0, spam=1).

2. **Split data**
   - Train/validation/test splits with fixed `random_state`.

3. **Preprocess + TF‑IDF**
   - Use `TfidfVectorizer` with:
     - `lowercase=True`
     - `stop_words='english'`
     - optional `min_df`, `max_df` to reduce noise
   - Fit on training text, transform val/test text.

4. **Train Logistic Regression**
   - Use `LogisticRegression(max_iter=...)`.
   - Train on TF‑IDF features.

5. **Train MLP (Keras)**
   - Convert sparse TF‑IDF matrix to dense or use `tf.keras` with sparse input support.
   - Small MLP: Input → Dense(64, ReLU) → Dense(1, Sigmoid).
   - Compile with `binary_crossentropy` and `adam`.

6. **Evaluate and compare**
   - Compute accuracy, precision, recall, F1 for both models on the test set.

7. **Top 10 most informative words**
   - For Logistic Regression:
     - Get coefficients `clf.coef_[0]`.
     - Sort by absolute magnitude.
     - Map indices to `vectorizer.get_feature_names_out()`.
     - Report top 10 words.

### Outputs to include
- Metrics table for Logistic Regression vs MLP
- List of top 10 informative words

---

## Deliverable 7: Replace TF‑IDF with Embeddings

### Goal
Replace TF‑IDF features with embeddings (e.g., spaCy or Word2Vec) and evaluate whether performance changes; hypothesize why.

### Process
1. **Choose embedding approach**
   - **spaCy**: use a model with vectors (e.g., `en_core_web_md`).
   - **Word2Vec/GloVe**: load pretrained vectors and average word vectors per document.

2. **Build document embeddings**
   - Tokenize each SMS.
   - For each text, compute an average of word vectors.
   - Result: fixed‑length embedding per document.

3. **Train models on embeddings**
   - Logistic Regression (baseline).
   - Small MLP (same size as before, but input is embedding dimension).

4. **Evaluate and compare**
   - Compute accuracy, precision, recall, F1 on test set.
   - Compare against TF‑IDF results.

5. **Hypothesis**
   - Embeddings may capture semantic similarity better and improve generalization.
   - But averaging embeddings can lose word order and nuance.
   - If dataset is small or domain‑specific, TF‑IDF might still outperform embeddings.

### Outputs to include
- Metrics table for embeddings vs TF‑IDF
- Short hypothesis on performance differences

---

## Notes for Implementation
- Use a fixed `random_state` for reproducibility.
- Keep evaluation consistent across models and feature types.
- Report metrics using `sklearn.metrics` for all models.
- Use clean, minimal dependencies (scikit‑learn, tensorflow/keras, numpy, pandas, matplotlib, spaCy or gensim if embeddings).

