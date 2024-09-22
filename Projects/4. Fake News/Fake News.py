# %%
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

# %%
news_data=pd.read_csv('news.csv')

# %%
news_data.shape

# %%
news_data.head()

# %%
news_data.isnull().sum()

# %%
news_data=news_data.fillna('')

# %%
news_data['content']=news_data['author']+' '+news_data['title']

# %%
X=news_data.drop(columns='label',axis=1)
Y=news_data['label']

# %%
#Stemming
port_stem=PorterStemmer()

# %%
def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)
  return stemmed_content

# %%
news_data['content']=news_data['content'].apply(stemming)

# %%
X=news_data['content'].values
Y=news_data['label'].values

# %%
#text to numbers
vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)

# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

# %%
model=LogisticRegression()

# %%
model.fit(X_train,Y_train)

# %%
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)

# %%
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)

# %%
#Predictive System
X_new=X_test[4]
prediction=model.predict(X_new)
print(prediction)
if(prediction[0]==0):
  print("This is a Real News")
else:
  print("This is a Fake News")

# %%



