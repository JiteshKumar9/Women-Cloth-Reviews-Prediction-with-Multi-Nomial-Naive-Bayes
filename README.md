# Women Cloth Reviews Prediction with Multinomial Naive Bayes

This project involves predicting customer reviews of women’s clothing using a Multinomial Naive Bayes classifier. The dataset consists of customer reviews, and the goal is to predict whether the review rating is "Good" (4 or 5) or "Poor" (1, 2, or 3).

## Dataset

The dataset used for this project is the **Women’s Clothing E-Commerce Reviews** dataset, which contains the following columns:

- `Clothing ID`: The ID of the clothing item.
- `Age`: Age of the reviewer.
- `Title`: Title of the review.
- `Review Text`: The text of the review.
- `Rating`: Rating of the product (1 to 5).
- `Recommended IND`: Whether the reviewer recommends the product (1 or 0).
- `Positive Feedback Count`: Number of positive feedbacks received for the review.
- `Division Name`: Division name of the product.
- `Department Name`: Department name of the product.
- `Class Name`: Class name of the product.

## Objective

The goal of this project is to predict whether the review rating is **Good** (4 or 5) or **Poor** (1, 2, or 3). This binary classification problem is solved using the **Multinomial Naive Bayes** classifier.

## Steps Involved

1. **Data Preprocessing**:
   - Load the dataset.
   - Handle missing values in the `Review Text` column by replacing them with "No Review Text".
   - Convert the `Rating` column into binary values: `0` for poor reviews (ratings 1, 2, 3) and `1` for good reviews (ratings 4, 5).

2. **Text Feature Extraction**:
   - Use `CountVectorizer` to convert the review text into a matrix of token counts (bigrams and trigrams).

3. **Model Training**:
   - Split the data into training and testing sets.
   - Train a **Multinomial Naive Bayes** model on the training set.

4. **Model Evaluation**:
   - Predict the ratings on the test set.
   - Evaluate the model using **confusion matrix** and **classification report**.

## Installation

To run this project locally, you'll need to install the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn


**Code Overview**
The code is structured as follows:

Import Libraries: Import the necessary libraries for data manipulation, visualization, and model training.
Data Preprocessing: Handle missing values, clean the dataset, and convert ratings into binary values.
Text Vectorization: Convert the review text into a matrix of token counts using CountVectorizer.
Model Training: Train the Multinomial Naive Bayes model on the training data.
Model Evaluation: Evaluate the model's performance using the confusion matrix and classification report.

**Results**
The final model achieved the following performance on the test set:
Accuracy: 69%
Precision: 0.64 (weighted average)
Recall: 0.69 (weighted average)
F1-Score: 0.67 (weighted average)


**Confusion Matrix:**
[[ 196 1387]
 [ 768 4695]]


```Classification Report:


              precision    recall  f1-score   support

         0.0       0.20      0.12      0.15      1583
         1.0       0.77      0.86      0.81      5463

    accuracy                           0.69      7046
   macro avg       0.49      0.49      0.48      7046
weighted avg       0.64      0.69      0.67      7046

**Conclusion**
The Multinomial Naive Bayes classifier provides a reasonable solution for predicting customer reviews as either "Good" or "Poor". The model shows a good balance between precision and recall for the "Good" reviews, although there is room for improvement in predicting "Poor" reviews.

