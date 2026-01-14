# NLP_Sentiment_Analysis_Assignment
Using Python to meet the minimum project requirements:

  1) Import the necessary libraries, including 'pandas', 'CountVectorizer' from 'sklearn.feature_extraction.text', 'train_test_split' from 'sklearn.model_selection', 'SVC' from 'sklearn.svm', and 'accuracy_score' from 'sklearn.metrics'.

  2) Load the dataset from a CSV file using 'pd.read_csv()'.

  3) Select the columns containing the text data and sentiment labels.

  4) Split the dataset into features (X) and target (y) using the selected columns.

  5) Split the data into training and testing sets using 'train_test_split()', specifying the test size and random seed.

  6) Vectorize the text data using 'CountVectorizer', fitting and transforming the training data and transforming the testing data.

  7) Initialize the SVM classifier using 'SVC()'.

  8) Train the SVM model using the vectorized training data and corresponding sentiment labels.

  9) Make predictions on the testing set using the trained model and the vectorized testing data.

  10) Evaluate the model's performance by calculating the accuracy score using 'accuracy_score()' and the true and predicted labels.

  11) Print the accuracy of the model using 'print()'.
