from numpy import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from methods import *

from todense import  DenseTransformer
import preprocessing

from time import time

from pprint import pprint

train = preprocessing.load_train_data()
test = preprocessing.load_test_data()

cm=ClassifyMethods(train,test)
cm.do2()
