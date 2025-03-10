#  $${\color{blue}Student \space Loans \space Default \space Prediction}$$

## Project Summary
> This project aims to build an ML model to predict whether a student loan will default. Using a dataset of student loans, the model is trained to understand patterns and factors that contribute to loan defaults. This prediction can help financial institutions and students make more informed decisions.

## Requirements
- Python 3.7+
- TensorFlow 2.0+
- scikit-learn
- pandas
- numpy
- matplotlib

## $${\color{blue}Project \space Steps: }$$

**Step 1: Data Preprocessing**
In this step, the dataset is loaded and cleaned by handling missing values and encoding categorical variables. We also split the data into training and testing sets.
```
y = loans_df["credit_ranking"]
X = loans_df.drop(columns="credit_ranking")
x_train, x_test, y_train, y_test = train_test_split(X,y)
```
>> x is defined by dropping the credit_ranking column from the DataFrame. The resulting DataFrame x contains all the columns except credit_ranking, which will be used as features for modeling or analysis.

**Step 2: Build the Model**
We create a neural network model using TensorFlow and Keras. The model architecture is defined, including the input layer, hidden layers, and output layer. The model is then compiled with an appropriate loss function and optimizer.

**Step 3: Train the Model**
The training data is used to train the model. During this step, the model learns to identify patterns and factors that contribute to loan defaults.

**Step 4: Evaluate the Model**
The model's performance is evaluated using the test data. We calculate the model’s loss and accuracy to determine how well it predicts loan defaults.

```
Model Loss: 0.5059076547622681
Accuracy Metric: 0.7674999833106995
```

**Step 5: Save the Model**
After evaluating the model, we save it to a file named student_loans.keras. This allows us to reuse the model without retraining it.

**Step 6: Classification Report**
Finally, we generate a classification report using the test data and the model's predictions. This report provides detailed metrics on the model's performance, including precision, recall, and F1-score.

![](classification_report_image.PNG)

### $${\color{blue}The \space classification \space report \space provides \space key \space metrics \space for \space evaluating \space the \space performance \space of \space a \space classification \space model}$$

**Precision: Measures the accuracy of the positive predictions.**
> For class 0, it's 0.71, meaning 71% of the predicted 0s were correct. For class 1, it's 0.83, meaning 83% of the predicted 1s were correct.

**Recall: Measures how well the model identifies all positive instances.**
> For class 0, it's 0.82, indicating the model correctly identified 82% of the actual 0s. For class 1, it's 0.72, meaning it correctly identified 72% of the actual 1s.

**F1-score: The harmonic mean of precision and recall.**
> It's 0.76 for class 0 and 0.77 for class 1, indicating a balanced performance between precision and recall.

**Support: The number of actual instances in each class.**  
> There are 180 instances of class 0 and 220 of class 1.

Overall, the model has an accuracy of 0.77, meaning it correctly predicted 77% of the total instances. The macro average and weighted average F1-scores are also 0.77, indicating consistent performance across both classes. The macro average treats all classes equally, while the weighted average takes the support (number of instances) of each class into account.
