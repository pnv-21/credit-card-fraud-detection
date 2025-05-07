# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions from a highly imbalanced real-world dataset, where fraud cases make up less than 0.2% of all transactions. The objective is to minimize false negatives while maintaining strong precision using interpretable machine learning models.

## Key Techniques

- **EDA**: Analyzed transaction time, amount, and PCA features (V1–V28)
- **Feature Engineering**: Added binary flags for suspicious V-feature ranges
- **Preprocessing**: Normalized features using `StandardScaler`
- **Imbalance Handling**: Used `class_weight='balanced'` to address skewed class distribution
- **Modeling**: Trained Logistic Regression and Random Forest classifiers
- **Evaluation**: Measured precision, recall, F1-score, AUC, and confusion matrices

## Results

| Model              | Recall (Fraud) | Precision (Fraud) | AUC   |
|-------------------|----------------|-------------------|-------|
| Logistic Regression | 0.91           | 0.50              | 0.97  |
| Random Forest       | 0.74           | 0.97              | 0.95  |

- Logistic Regression minimized false negatives with over 91% recall
- Random Forest provided high fraud precision (97%) with fewer false alarms

## Technologies Used

Python  
Pandas  
Scikit-learn  
Logistic Regression
Random Forest
Matplotlib  
Seaborn
