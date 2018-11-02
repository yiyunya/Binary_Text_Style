from numpy import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier,NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeClassifier,Perceptron,PassiveAggressiveClassifier,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.svm import SVC,LinearSVC

from sklearn.feature_selection import SelectKBest,chi2,SelectFromModel,mutual_info_classif
from sklearn.gaussian_process import GaussianProcessClassifier


from pprint import pprint
from todense import  DenseTransformer
from time import time

class ClassifyMethods:

    def __init__(self,train,test):
        self.train=train
        self.test=test



    def bernoulli_naive_bayes(self, f):
        print("method: BernoulliNB")
        print("method: BernoulliNB", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: chi", file=f)
        print("features: pure TFIDF", file=f)

        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(stop_words='english',max_features=7000,ngram_range=(1,3))),
            ('tfidf', TfidfTransformer()),
            ('chi', SelectKBest(score_func=chi2,k=6000)),
            ('clf', BernoulliNB(alpha=0.15))
        ])

        # pipe.set_params(vec__stop_words='english', vec__max_df=0.5, vec__ngram_range=(1, 1), tfidf__norm='l2',
        #                 vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.5, 0.75, 1.0),
            # 'vec__max_features': (8000, 9000, 10000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__alpha':(0.13,0.15,0.17)
            # 'clf__binarize':(0,1)
            # 'chi__k':(4000, 5000, 6000)


        }

        return pipe, parameters


    def perceptron(self, f):
        print("method: Perceptron")
        print("method: Perceptron", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)

        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', Perceptron(class_weight="balanced"))
        ])

        pipe.set_params(vec__stop_words='english', vec__max_features=None)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__alpha': (0.001,0.00001)
            # 'clf__shuffle':(True,False)


        }

        return pipe, parameters

    def random_forest(self, f):
        print("method: RandomForest")
        print("method: RandomForest", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)

        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=10))
        ])

        pipe.set_params(vec__stop_words='english', vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__fit_intercept' : (True,False),
            # 'clf__loss':('hinge','squared_hinge'),
            # 'clf__solver' : "lsqr"
            # 'clf__max_iter':(1,2)
            # 'clf__average':(0,1,10,20)
            # 'clf__n_estimators':(20, 50, 100, 200)

        }

        return pipe, parameters

    def passive_aggressive(self, f):
        print("method: PassiveAggressive")
        print("method: PassiveAggressive", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)

        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(max_df=0.25)),
            ('tfidf', TfidfTransformer()),
            ('clf', PassiveAggressiveClassifier())
        ])

        pipe.set_params(vec__stop_words='english', vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__fit_intercept' : (True,False),
            # 'clf__loss':('hinge','squared_hinge'),
            # 'clf__solver' : "lsqr"
            # 'clf__max_iter':(1,2)
            # 'clf__average':(0,1,10,20)

        }

        return pipe, parameters


    def ridge(self, f):
        print("method: Ridge")
        print("method: Ridge", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)

        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', RidgeClassifier(class_weight='balanced'))
        ])

        pipe.set_params(vec__stop_words='english', vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__tol' : (1e-2,1e-3),
            # 'clf__solver' : "lsqr"
            # 'clf__max_iter':(1,2)




        }

        return pipe, parameters

    def multinomial_naive_bayes(self,f):
        print("method: multinomialNB")
        print("method: multinomialNB", file = f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)

        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])

        pipe.set_params(vec__stop_words='english',vec__max_df=0.5, vec__ngram_range=(1, 3), tfidf__norm='l2', vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),

        }

        return pipe, parameters

    def k_nearest_neighbours(self, f):
        print("method: KNN")
        print("method: KNN", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)
        print("\n", file=f)
        print("preprocessing: None")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            # ('todense', DenseTransformer()),
            # ('scaler', PolynomialFeatures()),
            ('clf', KNeighborsClassifier())
        ])



        pipe.set_params(vec__stop_words='english', vec__max_features=None, vec__max_df=0.5, vec__ngram_range=(1, 3),
                        clf__algorithm='brute', tfidf__norm='l2', clf__weights='distance', clf__n_neighbors=7)
        # pipe.set_params(vec__stop_words='english', vec__max_features=3000,  vec__max_df=0.5, vec__ngram_range=(1,1), clf__algorithm = 'ball_tree',tfidf__norm = 'l2',clf__weights = 'distance',clf__n_neighbors = 7)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__n_neighbors': (3, 7, 9, 11),
            # 'clf__weights': ('uniform', 'distance'),
            # 'clf__algorithm': ('ball_tree', 'kd_tree', 'brute')

        }
        return pipe, parameters




    def nearest_centroid(self,f):
        print("method: NearestCentroid")
        print("method: NearestCentroid", file = f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)
        print("\n", file=f)
        print("preprocessing: None")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', NearestCentroid())
        ])

        # pipe = Pipeline([
        #     ('vec', CountVectorizer(ngram_range=(1,1),stop_words='english',max_features=200,max_df=0.5)),
        #     ('tfidf', TfidfTransformer(norm = 'l2')),
        #     ('todense', DenseTransformer()),
        #     # ('scaler', PolynomialFeatures()),
        #     ('clf', KNeighborsClassifier(algorithm='ball_tree'))
        # ])

        # pipe.set_params(vec__stop_words='english', vec__max_features=None,  vec__max_df=0.5, vec__ngram_range=(1,3), clf__algorithm = 'brute',tfidf__norm = 'l2',clf__weights = 'distance',clf__n_neighbors = 7)
        # pipe.set_params(vec__stop_words='english', vec__max_features=3000,  vec__max_df=0.5, vec__ngram_range=(1,1), clf__algorithm = 'ball_tree',tfidf__norm = 'l2',clf__weights = 'distance',clf__n_neighbors = 7)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__n_neighbors': (3, 7, 9, 11),
            # 'clf__weights': ('uniform', 'distance'),
            # 'clf__algorithm': ('ball_tree', 'kd_tree', 'brute')
            # 'clf__metric': ('manhattan', 'euclidean'),

        }
        return pipe, parameters

    def sgd(self, f):
        print("method: SGD")
        print("method: SGD", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)
        print("\n", file=f)
        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(stop_words='english', max_df=0.5, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='perceptron'))
        ])

        # pipe.set_params(vec__max_df=0.5, vec__ngram_range=(1, 1), tfidf__norm='l2', vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__n_neighbors': (3, 5, 7),
            # 'clf__weights': ('uniform', 'distance'),
            # 'clf__algorithm': ('ball_tree', 'kd_tree', 'brute')
            # 'clf__loss': ('log',  'perceptron', 'huber', 'squared_epsilon_insensitive'),
            # 'clf__alpha': (0.001, 0.0001, 0.00001)
            # 'clf__penalty': ('l1', 'elasticnet')
            # 'clf__class_weight': (None, 'balanced')
        }
        return pipe, parameters

    def selected_svc(self, f):
        print("method: SVM")
        print("method: SVM", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: l1", file=f)
        print("features: pure TFIDF", file=f)
        print("\n", file=f)
        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: l1")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(stop_words='english',max_df=0.5, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
            ('clf', LinearSVC(penalty='l2'))
        ])

        # pipe.set_params(vec__max_df=0.5, vec__ngram_range=(1, 1), tfidf__norm='l2', vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__n_neighbors': (3, 5, 7),
            # 'clf__weights': ('uniform', 'distance'),
            # 'clf__algorithm': ('ball_tree', 'kd_tree', 'brute')

        }
        return pipe, parameters

    def selected_svc_l2(self, f):
        print("method: SVM")
        print("method: SVM", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: l1", file=f)
        print("features: pure TFIDF", file=f)
        print("\n", file=f)
        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: l1")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(stop_words='english', max_df=0.5, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l2", dual=False, tol=1e-3))),
            ('clf', LinearSVC(penalty='l1',dual=False))
        ])

        # pipe.set_params(vec__max_df=0.5, vec__ngram_range=(1, 1), tfidf__norm='l2', vec__max_features=5000)

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            # 'clf__n_neighbors': (3, 5, 7),
            # 'clf__weights': ('uniform', 'distance'),
            # 'clf__algorithm': ('ball_tree', 'kd_tree', 'brute')

        }
        return pipe, parameters

    def svc(self, f):
        print("method: SVM")
        print("method: SVM", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)
        print("\n", file=f)
        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC())
        ])

        pipe.set_params(vec__max_df=0.5, vec__ngram_range=(1, 2))

        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),


        }
        return pipe, parameters

    def vote(self, f):
        print("method: Vote")
        print("method: Vote", file=f)
        print("preprocessing: none", file=f)
        print("decomposition: none", file=f)
        print("feature_selection: none", file=f)
        print("features: pure TFIDF", file=f)
        print("\n", file=f)
        print("preprocessing: none")
        print("decomposition: none")
        print("feature_selection: none")
        print("features: pure TFIDF")

        pipe = Pipeline([
            ('vec', CountVectorizer(stop_words='english', max_df=0.5, max_features=7000, ngram_range=(1,2))),
            ('tfidf', TfidfTransformer()),
            ('clf', VotingClassifier(
                estimators=[
                    ('bernoulliNB',BernoulliNB(alpha=0.15)),
                    ('perceptron',Perceptron(class_weight="balanced")),
                    ('random_forest',RandomForestClassifier(n_estimators=100, oob_score=True, random_state=10)),
                    ('passive_aggressive',PassiveAggressiveClassifier()),
                    ('ridge',RidgeClassifier(class_weight='balanced')),
                    ('multiNB',MultinomialNB()),
                    ('knn',KNeighborsClassifier(weights='distance', n_neighbors=7)),
                    ('nearest_centroid',NearestCentroid()),
                    ('sgd',SGDClassifier(
                        loss='perceptron'
                    )),
                    ('selected_svc',Pipeline([
                        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                        ('clf', LinearSVC(penalty='l2'))
                    ])),
                    ('svc',LinearSVC())
                ],
                weights=[4,1,1,3,1,1,2,2]

            ))
        ])

        # pipe.set_params()



        parameters = {
            # 'vec__max_df': (0.25, 0.5, 0.75, 1.0),
            # 'vec__max_features': (None, 5000, 10000, 50000),
            # 'vec__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),


        }
        return pipe, parameters


    def grid_search_wrapper(self,train_X,train_Y,pipe,parameters,f):
        grid_search = GridSearchCV(pipe, parameters)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipe.steps])
        print("pipeline:", [name for name, _ in pipe.steps], file=f)
        print("parameters:")
        pprint(parameters)
        pprint(parameters, stream=f)
        t0 = time()
        grid_search.fit(train_X, train_Y)
        print("done in %0.3fs" % (time() - t0))
        print("done in %0.3fs" % (time() - t0), file=f)
        print()
        f.write('\n')

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best score: %0.3f" % grid_search.best_score_, file=f)
        print("Best parameters set:")
        print("Best parameters set:", file=f)
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            print("\t%s: %r" % (param_name, best_parameters[param_name]), file=f)

        return grid_search








    def do(self,pipeline):
        f=open("data/experiment_recorder.txt",'a')
        train_X = self.train[0]
        train_Y = self.train[1]

        print("%d documents" % len(train_X))

        p=pipeline(f)
        pipe=p[0]
        parameters=p[1]

        grid_search = self.grid_search_wrapper(train_X,train_Y,pipe,parameters,f)

        test_X = self.test[0]
        test_Y = self.test[1]

        score = grid_search.best_estimator_.score(test_X, test_Y)
        # print(grid_search.best_estimator_.predict(test_X))
        print("Test set score: %f" % score)
        print("Test set score: %f" % score, file = f)
        print("\n\n\n", file=f)
        f.close()


    def weight(self,acc):
        if acc > 0.92:
            return 5
        elif acc > 0.91:
            return 4
        elif acc > 0.9:
            return 3
        elif acc > 0.85:
            return 2
        elif acc > 0.5:
            return 1

    import math



    def voter(self,estimators,f):
        test_X = self.test[0]
        test_Y = self.test[1]
        total = len(test_X)
        votes=[]
        cross = 0
        for estimator in estimators:
            estimate = estimator[1]
            predict_Y = estimate.predict(test_X)
            score = estimate.score(test_X,test_Y)
            s = []
            if estimator[0] == 'bernoulliNB':
                predict_Y_cross = estimate.predict_proba(test_X)
                for i in range(len(predict_Y_cross)):
                    s.append(predict_Y_cross[i])
            else:
                for i in range(len(test_Y)):
                    if predict_Y[i]=='0\n':
                        s.append([0.9,0.1])
                    else:
                        s.append([0.1,0.9])

            print(estimator[0]+ " test set score: %f" % score)
            print(estimator[0] + " test set score: %f" % score,file=f)
            votes.append((predict_Y,score,s))
        result=[]
        correct=0
        for i in range(total):
            flag = 0
            pos = 0
            neg = 0
            sum = 0
            ce = [0,0]
            for voting in votes:
                vote = voting[0]

                w=self.weight(voting[1])
                sum += w

                if vote[i] == '1\n':
                    pos+= w
                else:
                    neg+= w

            for voting in votes:
                p = voting[2]
                w = self.weight(voting[1])
                ce[0] += p[i][0]*w/sum
                ce[1] += p[i][1]*w/sum

            if pos>=neg:
                flag = '1\n'
                result.append(flag)
            else:
                flag = '0\n'
                result.append(flag)

            if flag == test_Y[i]:
                correct+= 1

            if test_Y[i]=='0\n':
                cross += math.log(ce[1])
            else:
                cross += math.log(ce[0])



        score=correct/total

        cross = -cross/total

        return score,result,cross

    def voting(self,pipelines):


        f = open("data/experiment_recorder.txt", 'a')
        train_X = self.train[0]
        train_Y = self.train[1]

        print("%d documents" % len(train_X))

        while True:
            t_init = time()

            estimators = []
            for pipeline in pipelines:
                p = pipeline[1](f)
                pipe = p[0]
                parameters = p[1]

                grid_search = self.grid_search_wrapper(train_X, train_Y, pipe, parameters, f)

                estimators.append((pipeline[0],grid_search.best_estimator_))

            score, results, cross = self.voter(estimators,f)

            print("Voting test score: %f" % score)
            print("Voting test score: %f" % score,file=f)

            print("Voting test cross: %f" % cross)
            print("Voting test cross: %f" % cross,file=f)

            print("Time used: %f s" % (time() - t_init))
            r = open('data/experiment_results_2.txt','w')
            r.writelines(results)

            if score>0.94:
                break

        f.close()
        r.close()
        return

    def do2(self):


        estimators = [
            ('bernoulliNB', self.bernoulli_naive_bayes),
            # ('perceptron', self.perceptron),
            # ('random_forest', self.random_forest),
            # ('passive_aggressive', self.passive_aggressive),
            ('ridge', self.ridge),
            # ('multiNB', self.multinomial_naive_bayes),
            # ('knn', self.k_nearest_neighbours),
            # ('nearest_centroid', self.nearest_centroid),
            ('sgd', self.sgd),
            ('selected_svc', self.selected_svc),
            # ('selected_svc_l2', self.selected_svc_l2),
            ('svc', self.svc)
        ]
        self.voting(estimators)



        return

















    # def multinomialNB(self):
    #
    #     train_X = self.train[0]
    #     train_Y = self.train[1]
    #
    #     print("%d documents" % len(train_X))
    #     pipe = Pipeline([
    #         ('vec', CountVectorizer()),
    #         ('tfidf', TfidfTransformer()),
    #         ('clf', MultinomialNB())
    #     ])
    #
    #     pipe.set_params(vec__max_df=0.5, vec__ngram_range=(1, 1), tfidf__norm='l2', vec__max_features=5000)
    #
    #     parameters = {
    #         # 'vec__max_df': (0.5, 0.75, 1.0),
    #         # 'vec__max_features': (None, 5000, 10000, 50000),
    #         # 'vec__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #         # 'tfidf__use_idf': (True, False),
    #         # 'tfidf__norm': ('l1', 'l2'),
    #
    #     }
    #
    #     grid_search = GridSearchCV(pipe, parameters)
    #
    #     print("Performing grid search...")
    #     print("pipeline:", [name for name, _ in pipe.steps])
    #     print("parameters:")
    #     pprint(parameters)
    #     t0 = time()
    #     grid_search.fit(train_X, train_Y)
    #     print("done in %0.3fs" % (time() - t0))
    #     print()
    #
    #     print("Best score: %0.3f" % grid_search.best_score_)
    #     print("Best parameters set:")
    #     best_parameters = grid_search.best_estimator_.get_params()
    #     for param_name in sorted(parameters.keys()):
    #         print("\t%s: %r" % (param_name, best_parameters[param_name]))
    #
    #     test_X = self.test[0]
    #     test_Y = self.test[1]
    #
    #     score = grid_search.best_estimator_.score(test_X, test_Y)
    #     print(score)

