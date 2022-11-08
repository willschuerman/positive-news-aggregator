

# %%
import pandas as pd
import os
from ast import literal_eval

# Supervised learning
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# %% 
# Import data
data_dir = os.getcwd().replace('src/models','data/interim/')
data = pd.read_csv(data_dir + 'abcnews_labeled.csv',converters={'Comments':literal_eval})

#Filtering out unlabeled data points
data= data.loc[data.label.isin([0,1]), :]
# %%
def text_representation(data):
  tfidf_vect = TfidfVectorizer()
  data['text'] = data['text'].apply(lambda text: " ".join(set(eval(text))))
  X_tfidf = tfidf_vect.fit_transform(data['text'])
  #print(X_tfidf.shape)
  #print(tfidf_vect.get_feature_names())
  X_tfidf = pd.DataFrame(X_tfidf.toarray())
  return X_tfidf
#apply the TFIDV function
X_tfidf = text_representation(data)

# %%
X= X_tfidf
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#fit Log Regression Model
clf= LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# Save model for subsequent runs



# %%
new_data = ["The US imposes sanctions on Rassia because of the Ukranian war"]
tf = TfidfVectorizer()
tfdf = tf.fit_transform(data['text'])
vect = pd.DataFrame(tf.transform(new_data).toarray())
new_data = pd.DataFrame(vect)
logistic_prediction = clf.predict(new_data)
print(logistic_prediction)