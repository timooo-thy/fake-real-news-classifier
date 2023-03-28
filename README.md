# Fake/Real News Detection
News Classification (Dataset from Kaggle)
# About
Our project is all about fighting fake news by building a system that can automatically tell if a news article is real or fake. In today's world of social media and online news, it's becoming harder to know what to believe. That's where our machine learning model comes in - it analyses the content of news articles using fancy language processing techniques to figure out if they're telling the truth or not.

To address this problem, we have built a machine learning model that uses natural language processing techniques to analyse the content of news articles and determine their credibility. We trained our model on a large dataset of labeled news articles, which we carefully curated to include examples of both real and fake news.

Our model uses a variety of features, such as the text and title, the length of the article, and the overall tone and sentiment, to make its classification decision. We also experimented with different machine learning algorithms and techniques, such as PyTorch and Keras, to improve the accuracy and robustness of our system. 

We're hoping that our fake news classification system will be really useful for anyone who wants to stay on top of what's happening in the world without getting duped by fake news. It's gonna be a big help for journalists, fact-checkers, and anyone else who wants to know what's really going on out there.

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
3) Random Forest with Cross Validation
4) Pytorch using Bert Based Uncased Model (model not included in github due to large file size)
5) Keras ANN using Tokenizer for preprocessing text (model included)
6) XGBoost Deep Learning using TF-IDF to vectorise text (model included) with Cross Validation

# Further Testing (on an unseen dataset)
We tested our models (keras and xgboost) on a completely new dataset to test its perfomance against real world news.

# Conclusion

# Takeaways

# References
1) https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2) https://huggingface.co/bert-base-uncased
3) https://keras.io/api/keras_nlp/tokenizers/tokenizer/
4) https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5) https://towardsdatascience.com/a-simple-explanation-of-the-bag-of-words-model-b88fc4f4971
6) https://monkeylearn.com/sentiment-analysis/
7) https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
