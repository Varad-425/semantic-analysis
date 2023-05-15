# import statements
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class type_Tweet():
    def __init__(self):
        nltk.download('stopwords')
        self.__data = pd.read_csv(
            'F:\CODING_AFTER_AUGUST\ML_Projects\sentiment_analysis\\twitter_training.csv')
        self.__cv = CountVectorizer(max_features=1500)
        self.__lr = LabelEncoder()
        self.__ps = PorterStemmer()
        self.__sc = StandardScaler()
        self.__all_stopwords = stopwords.words('english')
        self.__all_stopwords.remove('not')
        self.__corpus = []
        self.new_set = set(self.__data.iloc[:, 1])
        df = pd.DataFrame(self.new_set, columns=[
                          'Category'], index=range(1, len(self.new_set)+1))

    # CountVectorize here
    def get_counts(self, data):
        return self.__cv.transform(data).toarray()  # type: ignore

    # Removing stopword from the data
    def removing_Stopwords(self, data):
        review = re.sub('[^a-zA-Z]', ' ', str(data))
        review = review.lower()
        review = review.split()
        review = [self.__ps.stem(word) for word in review if not word in set(
            self.__all_stopwords)]
        review = ' '.join(review)
        self.__corpus.append(review)
        return review

    # Getting the categorical data
    def get_data(self, category):
        # 'Borderlands' is the column name for category column
        xds = self.__data[self.__data['Borderlands'] == category]
        X_ds = xds.iloc[:, 3].values
        Y_ds = xds.iloc[:, 2].values
        self.__Y = self.__lr.fit_transform(Y_ds)
        for i in range(len(X_ds)):
            temp_X = self.removing_Stopwords(X_ds[i])
        self.__cv.fit(self.__corpus)
        self.__X = self.get_counts(self.__corpus)

    # splitting dataset
    def split_data(self):
        self.__X_train, self.__X_test, self.__Y_train, self.__Y_test = train_test_split(
            self.__X, self.__Y, test_size=0.2, random_state=0)

    # Logistic Classification model
    def logistic_reg(self):
        self.__gb = LogisticRegression(random_state=0, max_iter=2000)
        self.__gb.fit(self.__X_train, self.__Y_train)

    # Prediction method by entering the tweet and return the result
    def prediction(self, data):
        print("*"*50)
        print('\n\n'+data[0]+'\n\n')
        list_data = self.removing_Stopwords(data)
        temp_list = []
        temp_list.append(list_data)
        list_data = self.get_counts(temp_list)
        list_data = self.standard_scale(list_data)
        pred = self.__lr.inverse_transform(self.__gb.predict(list_data))
        if (pred[0] == 'Irrelevant'):
            sttr = 'The tweet is Irrelevant to '+self.__categoory
        elif (pred[0] == 'Positive'):
            sttr = 'The tweet is a Positive statement to '+self.__categoory
        elif (pred[0] == 'Neutral'):
            sttr = 'The tweet is Neutral to '+self.__categoory
        else:
            sttr = 'The tweet is Negative statement to '+self.__categoory
        print(sttr+'\n\n')
        print("*"*50)
        return pred

    # Accuracy score of model dataset
    def get_accuracy(self):
        return accuracy_score(self.__Y_test, self.__gb.predict(self.__X_test))

    # Confusion matrix of model dataset
    def get_confusion_matrix(self):
        cm = confusion_matrix(self.__Y_test, self.__gb.predict(self.__X_test))
        return cm

    # Standard scaling of the dataset
    def standard_scale(self, data=None):
        if data is None:
            self.__X_train = self.__sc.fit_transform(self.__X_train)
            self.__X_test = self.__sc.transform(self.__X_test)
        else:
            data = self.__sc.transform(data)
            return data

    # this method is called where you enter a string category and will apply the model and then use prediction
    def run(self, category):
        self.get_data(category)
        self.split_data()
        self.standard_scale()
        self.logistic_reg()
        self.__categoory = category


# Creating object class
obj = type_Tweet()
print("*"*50, "\n\n")
while (True):
    # Enter the category from the list you get while running
    catg = str(input("Enter the category : "))
    if catg in obj.new_set:
        break
    else:
        print("Wrong input., Try again")


inputs = input("Enter the tweet : ")
# Primary use this method for setting up the model
obj.run(catg)

# Use for prediction of the tweet
pred = obj.prediction(inputs)
print(obj.get_accuracy())
print(obj.get_confusion_matrix())
