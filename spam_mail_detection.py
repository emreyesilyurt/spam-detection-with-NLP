

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


dataset = pd.read_csv('spam_ham_dataset.csv')
print(dataset.isnull().sum())


import re 

import nltk
stopWords = nltk.download('stopwords')

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


corpus = []
for i in range(len(dataset)):
    text = re.sub('[^a-zA-Z]',' ', dataset['text'][i])
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)

X = cv.fit_transform(corpus).toarray()      
y = dataset.iloc[:,-1:].values


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X, y,test_size=0.2, random_state=0)



from xgboost import XGBClassifier
model = XGBClassifier(n_estimators = 150, booster = 'gbtree')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('XGBoost Accuracy: ', metrics.accuracy_score(y_test, y_pred), '\n')
print("XGBClassifier Confusion Matrix\n", confusion_matrix(y_test, y_pred))

cross_val = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print('XGB Accuracy: ', cross_val.mean())
print('XGB Std: ', cross_val.std())





from sklearn.model_selection import GridSearchCV

params = [
    {'n_estimators' : [50,100,150,200, 250], 'booster' : ['gbtree', 'gblinear']}
    ]



gs = GridSearchCV(
    estimator = model,
    param_grid = params,
    scoring = 'accuracy',
    cv = 10,
    n_jobs = -1
    )



grid_search = gs.fit(X_train, y_train)
best_result = grid_search.best_score_
best_params = grid_search.best_params_

print('Best_Result', best_result)
print('Best_Params', best_params)


import pickle
save = pickle.dump(model, open('model.save', 'wb'))
