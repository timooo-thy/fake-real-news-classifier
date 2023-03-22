# Fake/Real News Detection
News Classification (Dataset from Kaggle)
# About

# Contributors
[@timooo-thy](https://github.com/timooo-thy)

# Problem Definition

# Cleaning Methods
1) Identify information leaks
  - Text leaks 
  - Date leaks
  - URL leaks

2) Basic NLP data cleaning
  - Contractions
  - Punctuations 
  - Spaces
  - Lowercase
  - Duplicates

# Analysis Done
1) NLTK and WordCloud analysis
2) Distribution of news article length (log transformation)
3) Sentiment analysis using TextBlob
4) Subject analysis

# Models Used
1) Logistic Regression (Sentiment Score and Bag of Words)
2) Binary Tree Classification
3) Random Forest
4) Pytorch using Bert Based Uncased Model (model not included in github due to large file size)
5) Keras ANN using Tokenizer for preprocessing text (model included)
6) XGBoost Deep Learning using TF-IDF to vectorise text (model included)
7) XGBoost 5-Fold Cross Validation

# Conclusion

# Takeaways

# References
1) https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2) https://huggingface.co/bert-base-uncased
3) https://keras.io/api/keras_nlp/tokenizers/tokenizer/
4) https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5) https://towardsdatascience.com/a-simple-explanation-of-the-bag-of-words-model-b88fc4f4971
6) https://monkeylearn.com/sentiment-analysis/
