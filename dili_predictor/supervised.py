import pandas as pd
import numpy as np
import warnings
import pickle
import time
import os
from pathlib import Path
from copy import deepcopy
from functools import wraps
from joblib import Parallel, delayed

from IPython.core.display import display
from scipy.spatial import distance_matrix

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from .feature_extractor import Data

__all__ = ['Model', 'RegressionModel', 'ClassificationModel']

MODELS = (
    {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'KernelRidge': KernelRidge(),
        'BayesianRidge': BayesianRidge(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'SVR_linear': SVR(),
        'SVR_rbf': SVR(),
        'SVR_poly': SVR(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'MLPRegressor': MLPRegressor(),
        #         'XGBRegressor': XGBRegressor()
    },
    {
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC_linear': SVC(),
        'SVC_rbf': SVC(),
        'SVC_poly': SVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'MLPClassifier': MLPClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'GaussianNB': GaussianNB(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }
)


def _valid_candidates(candidates, mode):
    '''Validate the ML algorithms.
    
    Parameters
    ----------
    candidates : str
        The ML algorithms will be tuned.
        
    mode : str
        Mode can be 'regression' or non-regression.
        
    Returns
    -------
    valid_candidates : dict
        The valid candidates.
        
    _MODEL:
        The keys and functions of ML algorithms.
    '''
    # model used for regression and classification
    valid_candidates = deepcopy(candidates)
    if mode == 'regression':
        _MODEL = MODELS[0]
    else:
        _MODEL = MODELS[1]
    for item in candidates:
        if item not in _MODEL:
            del valid_candidates[item]
            warnings.warn('%s is not a valid candidate. Use supervised.MODELS to get valid options.' % item)
    return valid_candidates, _MODEL


def _applicability_domain(x_train=None, x_test=None, knn=5, weight=1.4, verbose=True):
    '''applicability domain (AD) of the model.
    The AD threshold is defined as follow:
    AD threshold = Dk + weight * std

    Parameters
    ----------
    x_train, x_test : array-like
        The training and test data used to compute the AD threshold.

    knn : int, default=5
        The k-nearest neighbors used to calculate the Dk.

    weight : float, default=1.4
        An empirical parameters that ranges from 1 - 5.
        
    verbose : bool, default=True
        If true, print the AD results.
        
    Returns
    -------
    reliable : list
        The results of AD.
        Reliable means the sample within the AD.
        Unreliable means the sample is out of the AD.
    '''
    # calc the distance matrix of training set
    train_dis = distance_matrix(x_train, x_train)
    train_knn_dis = np.sort(train_dis, axis=1)[:, 1:knn + 1]

    # threshold
    train_mean_dis = np.mean(train_knn_dis, axis=None)
    std = np.std(train_knn_dis, axis=None)
    threshold = train_mean_dis + weight * std

    # calc the distance matrix of test set
    test_dis = distance_matrix(x_test, x_train)
    test_knn_dis = np.sort(test_dis, axis=1)[:, :knn]

    # calc AD
    test_mean_dis = np.mean(test_knn_dis, axis=1)
    test_threshold = test_mean_dis

    reliable = ['Reliable' if i < threshold else 'Unreliable' for i in test_threshold]
    selected = np.where(test_threshold <= threshold)[0]
    if verbose:
        print('Total test data :', x_test.shape[0])
        print('Unreliable data :', reliable.count('Unreliable'))
        print('Reliable data :', reliable.count('Reliable'))
    return reliable


class Model():
    '''This class receives the cleaned data and build ML models automatically.

    Parameters
    ----------
    x_train, y_train : array-like
        The training data and the corresponding labels.

    x_test, y_test : array-like
        The test data and the corresponding labels.
        
    predict_x, predict_y : array-like
        The new data or the external test data and the corresponding labels.

    data : Data object
        Instantiated object from ./feature_extractor.

    max_iter : int, default=1000
        Parameters for the ML models.

    n_jobs : int, deafault=-1
        Number of cpu that will be used.

    data_path : str, default='./data'
        The directory used to save the results.
    '''

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, predict_x=None, predict_y=None, data=None,
                 max_iter=5000, n_jobs=-1, data_path='./data'):

        # set the training and test data
        self.load_training_data(x_train, y_train, x_test, y_test, data)

        # set the predicted data
        self.load_predict_data(predict_x, predict_y, data)

        # set the data path
        if isinstance(data, Data):
            self.data_path = data.data_path
        else:
            self.data_path = Path(data_path)

        if n_jobs < 1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.data_path = Path('./data')
        self.max_iter = max_iter
        self.predict_df_res_dict = {}
        self.AD_dict = {}

    def load_training_data(self, x_train=None, y_train=None, x_test=None, y_test=None, data=None):
        '''Load the training data.

        Parameters
        ----------
        x_train, y_train : array-like
            The training data and the corresponding labels.

        x_test, y_test : array-like
            The test data and the corresponding labels.

        data : Data object.
            Instantiated object from ./feature_extractor.
        '''
        if isinstance(data, Data):
            self.df_x = data.df_x
            self.df_y = data.df_y
            self._train_data_exist = True
            self.label_col = data.label_col
            self.df_x_columns = data.df_x_columns
            if data._split:
                self.x_train = data.x_train
                self.y_train = data.y_train
                self.x_test = data.x_test
                self.y_test = data.y_test
                self._split = True
                print('%s training data and %s test data detected in Data!' % (
                    self.x_train.shape[0], self.x_test.shape[0]))
            else:
                self.x_train = data.df_x
                self.y_train = data.df_y
                self.x_test = data.df_x
                self.y_test = data.df_y
                self._split = False
                print('No test set detected in Data! Whole data set will be used as a test set.')
            print('%s features detected in Data!' % (self.x_train.shape[1]))

        elif (x_train is not None) and (y_train is not None):
            self.x_train = x_train
            self.y_train = y_train
            self._train_data_exist = True
            if (x_test is not None) and (y_test is not None):
                self.x_test = x_test
                self.y_test = y_test
                self.df_x = np.concatenate([x_train, x_test])
                self.df_y = np.concatenate([y_train, y_test])
                self._split = True
                print('%s training data and %s test data received!' % (self.x_train.shape[0], self.x_test.shape[0]))
            else:
                self.x_test = x_train
                self.y_test = y_train
                self.df_x = x_train
                self.df_y = y_train
                self._split = False
                print('No test set received! Whole data set will be used as a test set.')
            print('%s features detected in taining data!' % (self.x_train.shape[1]))
        #             self.df_x_columns = self.df_x.columns
        else:
            self._train_data_exist = False

    def load_predict_data(self, predict_x=None, predict_y=None, data=None):
        '''Load the predict data.

        Parameters
        ----------
        predict_x, predict_y : array-like
            The new data or the external test data and the corresponding labels.

        data : Data object.
            Instantiated object from ./feature_extractor.
        '''
        # set the predicted data
        if isinstance(data, Data):
            if data._predict_data_exist:
                self.predict_df = data.predict_df
                self.predict_df_y = data.predict_df_y
                self.predict_df_x = data.predict_df_x
                self.predict_label_col = data.predict_label_col
                self._predict_data_exist = True
            else:
                self._predict_data_exist = False
        elif predict_x is not None:
            self.predict_df_x = predict_x
            self.predict_df_y = predict_y
            self.predict_df = pd.concat([pd.DataFrame(predict_x), pd.DataFrame(predict_y)],
                                        columns=self.df_x_columns + ['label'])
            self.predict_label_col = 'label'
            self._predict_data_exist = True
        else:
            self._predict_data_exist = False

    def time_decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            time1 = time.time()
            res = func(*args, **kwargs)
            time2 = time.time()
            m, s = divmod((time2 - time1), 60)
            h, m = divmod(m, 60)
            print('{:=^70}\n'.format(' {} | Time: {}h {}min {:.0f}s '.format(func.__name__, h, m, s)))
            return res

        return wrap

    @time_decorator
    def model_tuning(self, scoring='neg_mean_squared_error', kfold_random_state=1, model_random_state=1):
        '''Tuning the parameters by GridSearchCV.
        
        Parameters
        ----------
        scoring : str or list, default='neg_mean_squared_error'
            Metrics to evaluate the model performance.
            
        kfold_random_state : int, default=1 
            Set the random_state for reproducibility of k-fold.
            
        model_random_state : int, default=1 
            Set the random_state for reproducibility of training model.
            
        Returns
        -------
        model_tuning_res : DataFrame
            All of the tuning results.
        '''
        self.scoring = scoring
        self.kfold_random_state = kfold_random_state
        self.model_random_state = model_random_state
        if self.__class__ == ClassificationModel:
            self.y_train = self.y_train.astype(str)
            self.y_test = self.y_test.astype(str)
        else:
            self.y_train = self.y_train.astype(float)
            self.y_test = self.y_test.astype(float)
        skf = KFold(n_splits=5, shuffle=True, random_state=kfold_random_state)
        self.KFold = skf
        candidates = self.candidates
        df_list = []
        for model_key in candidates:
            time1 = time.time()
            print('Tuning: ', model_key)
            
            # init estimator
            estimator = self._init_estimator(model_key)
            
            # tuning params
            model_params = candidates[model_key]

            # search the best parameters by GridSearchCV
            gsearch = GridSearchCV(estimator=estimator,
                                   param_grid=model_params,
                                   scoring=scoring,
                                   n_jobs=self.n_jobs,
                                   refit=False,
                                   cv=skf)
            gsearch.fit(self.x_train, self.y_train)
            df_model = pd.DataFrame(gsearch.cv_results_)
            df_model['algorithm'] = model_key
            df_list.append(df_model)
            time2 = time.time()
            print('Using time: %.2f\n' % (time2 - time1))

        self.model_tuning_res = pd.concat(df_list, ignore_index=True)
        self.model_tuning_res.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'], axis=1,
                                   inplace=True)
        return self.model_tuning_res

    @time_decorator
    def best_model_select(self, rank=None):
        '''Select the best parameters of each algorithm.
        
        Parameters
        ----------
        rank : str, default=None
            if two or more metrics were used in the tuning process,
            select one metrics used to rank the results.
            
        Returns
        -------
        best_df : DataFrame
            The best results selected by "rank".
        '''
        self.rank = rank
        res_df = self.model_tuning_res
        if isinstance(self.scoring, list):
            try:
                rank = 'rank_test_' + rank
                best_df = res_df[res_df[rank] == 1].copy(deep=True)
            except:
                raise Exception('select one of the metrics from %s' % self.scoring)
            else:
                self.display_cols = ['rank_test_' + i for i in self.scoring]
                self.display_cols.extend(['mean_test_' + i for i in self.scoring])
                self.display_cols.extend(['std_test_' + i for i in self.scoring])
        else:
            best_df = res_df[res_df['rank_test_score'] == 1].copy(deep=True)
            self.display_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']
        self.best_df = best_df
        self.display_cols.extend(['algorithm'])
        self.display_cols.extend(['params'])
        display(best_df[self.display_cols])
        return best_df

    def _init_estimator(self, model_key, **params):
        '''Initialize the estimator'''
        # set the model
        estimator = self._MODEL[model_key]
        if params:
            estimator.set_params(**params)

        # set the params
        if hasattr(estimator, 'max_iter'):
            estimator.max_iter = self.max_iter
        if hasattr(estimator, 'n_jobs'):
            estimator.n_jobs = self.n_jobs
        if hasattr(estimator, 'random_state'):
            estimator.random_state = self.model_random_state
        return estimator
    
    def _evaluate(self, index, alg, params, evaluate_metrics):
        # init estimator
        estimator = self._init_estimator(alg, **params)

        # train and predict
        estimator.fit(self.x_train, self.y_train)
        y_train_pred = estimator.predict(self.x_train)
        y_test_pred = estimator.predict(self.x_test)

        # add the results into dataframe
        res_list = []
        eval_dict = {
            'train_': [y_train_pred, self.y_train],
            'test_': [y_test_pred, self.y_test],
                    }
        for evaluate_metric in evaluate_metrics:
            for _, value in eval_dict.items():
                res = eval('metrics.%s' % evaluate_metric)(value[0], value[1])
                res_list.append(res)
        res_list.append(index)
        return res_list
    
    @time_decorator
    def all_model_evaluate(self, df, evaluate_metrics, verbose=True):
        '''Train on the training data and predict for the test data.
        
        Parameters
        ----------
        df / iter_df : DataFrame
            The tuning results returned by GridSearch, or other data sheet.
            
        evaluate_metrics : list, default=['mean_squared_error']
            The metrics to evaluate the model performance
            
        Returns
        -------
        df : DataFrame
            The data sheet with evaluation results.
        '''
        # reset index
        iter_df = df.copy()
        iter_df.reset_index(drop=True, inplace=True)
        
        # Out is a list of score
        out = Parallel(
            n_jobs=self.n_jobs, verbose=verbose
        )(
            delayed(self._evaluate)(index, row['algorithm'], row['params'], evaluate_metrics)
            for index, row in iter_df.iterrows())
        
        col_names = [
            prefix + m
            for m in evaluate_metrics
            for prefix in ['train_', 'test_']
        ]
        col_names.append('idx')
        df_res = pd.DataFrame(out, columns=col_names).set_index('idx')
        iter_df = pd.merge(iter_df, df_res, left_index=True, right_index=True)
        return iter_df
    
        
    @time_decorator
    def best_model_evaluate(self, evaluate_metrics=['mean_squared_error'], verbose=True):
        '''Evaluate for the best model.'''
        return self.all_model_evaluate(self.best_df, evaluate_metrics, verbose=verbose)
    
    @time_decorator
    def calc_ROC(self, pos_label=1, ax=None, display=None, label=None):
        '''Calc the AUC, SE, SP for the best model on the test set.'''
        res = []
        for index, row in self.best_df.iterrows():
            estimator = self._init_estimator(row['algorithm'], **row['params'])
            if hasattr(estimator, 'probability'):
                estimator.probability = True

            # train and predict
            estimator.fit(self.x_train, self.y_train)
            y_train_pred = estimator.predict(self.x_train)
            y_test_pred = estimator.predict(self.x_test)
            if hasattr(estimator, 'predict_proba'):
                y_test_pred_prob = estimator.predict_proba(self.x_test)

            fpr, tpr, thresholds = metrics.roc_curve(self.y_test, y_test_pred_prob[:, 1], pos_label=pos_label)
            auc = metrics.auc(fpr, tpr)
            acc = metrics.accuracy_score(self.y_test, y_test_pred)
            precision = metrics.precision_score(self.y_test, y_test_pred, pos_label=pos_label)
            recall = metrics.recall_score(self.y_test, y_test_pred, pos_label=pos_label)

            if display:
                display = metrics.plot_roc_curve(estimator, self.x_test, self.y_test, name=label, ax=display.ax_)
            else:
                display = metrics.plot_roc_curve(estimator, self.x_test, self.y_test, name=label, ax=ax)
            res.append([acc, auc, precision, recall])
        df_res = pd.DataFrame(res, columns=['ACC', 'AUC', 'precision', 'recall'])
        return df_res, display
        

    def _predict(self, index, alg, params):
        # init estimator
        estimator = self._init_estimator(alg, **params)

        # train and predict
        estimator.fit(self.df_x, self.df_y)
        predict_y = estimator.predict(self.predict_df_x)
        if hasattr(estimator, 'predict_proba'):
            predict_y_ = estimator.predict_proba(self.predict_df_x)

        predict_y = list(predict_y)
        predict_y.append('%s_%s'%(alg, index))
        return predict_y
    
    @time_decorator
    def all_model_predict(self, df, verbose=True):
        '''Train on the training data and predict for the test data.
        
        Parameters
        ----------
        df / iter_df : DataFrame
            The tuning results returned by GridSearch, or other data sheet.
            
        Returns
        -------
        df : DataFrame
            The data sheet with evaluation results.
        '''
        # reset index
        iter_df = df.copy()
        iter_df.reset_index(drop=True, inplace=True)
        
        # Out is a list of score
        out = Parallel(
            n_jobs=self.n_jobs, verbose=verbose
        )(
            delayed(self._predict)(index, row['algorithm'], row['params'])
            for index, row in iter_df.iterrows())
        
        # columns
        col_names = list(self.predict_df.index)
        col_names.append('alg')
        
        # combine
        df_res = pd.DataFrame(out, columns=col_names).T
        df_res.columns = df_res.iloc[-1]
        df_res.drop(df_res.index[-1], inplace=True)
        iter_df = pd.merge(self.predict_df, df_res, left_index=True, right_index=True)
        return iter_df
    
    
    @time_decorator
    def best_model_predict(self, verbose=True):
        '''Predict the unknown data using the best model and calc the metrics.'''
        return self.all_model_predict(self.best_df, verbose=verbose)
    

    def AD(self, knn=5, weight=1.4, verbose=True):
        '''An applicability domain is defined by knn and weight,
        which is used to determine how many samples within or out of AD.
        
        Parameters
        ----------
        knn : int, default=5
            The number of k-nearest neighbors.
            
        weight : float, default=1.4
            The weight of the standard deviation.
            
        verbose : bool
            Print the AD results
            
        Returns
        -------
        The number of samples within or out of AD.
        '''
        if len(self.predict_df_x) != self.predict_df_res.shape[0]:
            self.best_model_predict()
        ret = _applicability_domain(self.df_x, self.predict_df_x, knn, weight, verbose)
        self.predict_df_res['reliable'] = ret
        self.reliable = ret
        return ret.count('Unreliable'), ret.count('Reliable')

    def save_model(self, fname=None, ftime=True):
        '''save the object and results

        Parameters
        ----------
        fname : str, default=None
            The file name.

        time : bool, default=True
            Add the current time to the file name.
        '''
        if fname:
            fname = str(fname)
        if ftime:
            time_now = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        else:
            time_now = ''

        name_comp = [i for i in [fname, time_now] if i]
        self.save_file = self.data_path / '{}.pkl'.format('_'.join(name_comp))

        output = open(self.save_file, 'wb')
        pickle.dump(self, output)
        output.close()


class RegressionModel(Model):
    def set_candidates(self, candidates):
        '''Set the algorithms and parameters for the regression models.
        
        Parameters
        ----------
        candidates : dict
            The algorithm keys and the parameters will be tuned.
            eg. {'Lasso':{'alpha': [0.01, 0.1, 1]}
        '''
        if not candidates:
            candidates = {
                'LinearRegression':
                    {},
                'Lasso':
                    {
                        'alpha': np.logspace(-7, 3, 400)
                    },
                'Ridge':
                    {
                        'alpha': np.logspace(-7, 3, 400)
                    },
                'ElasticNet':
                    {
                        'l1_ratio': np.arange(0, 1.05, 0.05),
                        'alpha': np.logspace(-7, 3, 200),
                        #                     'tol': np.logspace(-4, 1, 200)
                    },
                'KernelRidge':
                    {
                        'alpha': np.logspace(-4, -2, 100),
                        'gamma': np.logspace(-5, -3, 100),
                        'kernel': ["laplacian", "sigmoid"]
                    },
                'BayesianRidge':
                    {
                        'n_iter': np.arange(300, 600, 50),
                        'tol': np.arange(1e-5, 1e-4, 50),
                        'alpha_1': np.arange(1e-9, 1e-3, 50),
                        'alpha_2': np.arange(1e-9, 1e-3, 50),
                        'lambda_1': np.arange(1e-9, 1e-3, 50),
                        'lambda_2': np.arange(1e-9, 1e-3, 50)
                    },
                'KNeighborsRegressor':
                    {
                        'n_neighbors': np.arange(2, 30, 2),
                        'weights': ['uniform', 'distance'],
                        #                     'leaf_size': np.arange(20, 40, 5)
                    },
                'SVR_linear':
                    {
                        'kernel': ['linear'],
                        'C': np.logspace(-7, 3, 400),
                    },
                'SVR_rbf':
                    {
                        'kernel': ['rbf', 'sigmoid'],
                        'C': np.logspace(-7, 3, 30),
                        'gamma': np.logspace(-7, 3, 30),
                        'epsilon': np.logspace(-7, 3, 30)
                    },
                'SVR_poly':
                    {
                        'kernel': ['poly'],
                        'degree': [2],
                        'C': np.logspace(-7, 3, 30),
                        'gamma': np.logspace(-7, 3, 30),
                        'epsilon': np.logspace(-7, 3, 30)
                    },
                'DecisionTreeRegressor':
                    {
                        'max_depth': np.arange(5, 13, 1),
                        'min_samples_split': np.arange(2, 11, 2),
                        'min_samples_leaf': np.arange(2, 11, 2),
                        'max_features': ['auto', 'sqrt', 'log2']
                    },
                'RandomForestRegressor':
                    {
                        'n_estimators': np.arange(50, 500, 100),
                        'min_samples_split': np.arange(2, 12, 3),
                        'min_samples_leaf': np.arange(2, 12, 3),
                        'max_depth': np.arange(5, 12, 2),
                        'max_features': ['sqrt', 'log2', 'auto']
                    },
                'GradientBoostingRegressor':
                    {
                        'n_estimators': np.arange(50, 600, 50),
                        'learning_rate': [0.01, 0.1, 1],
                        'min_samples_split': np.arange(2, 11, 3),
                        'min_samples_leaf': np.arange(1, 11, 3),
                        'max_depth': np.arange(5, 12, 2),
                        'max_features': ['sqrt', 'log2', 'auto']
                    },
                'AdaBoostRegressor':
                    {
                        'n_estimators': np.arange(50, 500, 50),
                        'learning_rate': np.linspace(0.05, 0.5, 1),
                        'loss': ['linear', 'square', 'exponential']
                    },
                'MLPRegressor':
                    {
                        'hidden_layer_sizes': [(2, 2), (2, 4), (2, 6), (2, 8),
                                               (4, 2), (4, 4), (4, 6), (4, 8),
                                               (6, 2), (6, 4), (6, 6), (6, 8), ],
                        'activation': ['tanh'],
                        'learning_rate_init': [0.001, 0.01, 0.1],
                        'alpha': np.logspace(-7, 2, 10),
                        'early_stopping': [True],
                        'max_iter': [1000]
                    },
                'XGBRegressor':
                    {
                        'gamma': np.arange(0, 0.5, 0.1),
                        'max_depth': np.arange(3, 11),
                        'min_child_weight': np.arange(1, 20),
                        'colsample_bytree': np.arange(0.5, 1, 0.1),
                        'subsample': np.arange(0.5, 1, 0.1),
                        'learning_rate': np.arange(0.001, 0.2, 0.05),
                        'n_estimators': np.arange(10, 500, 100)
                    },
            }
        self.candidates, self._MODEL = _valid_candidates(candidates, 'regression')


class ClassificationModel(Model):
    def set_candidates(self, candidates):
        '''Set the algorithms and parameters for the classification models.
        
        Parameters
        ----------
        candidates : dict
            The algorithm keys and the parameters will be tuned.
            eg. {'KNeighborsClassifier':{'n_neighbors': [5, 10, 15]}
        '''
        if not candidates:
            candidates = {
                'LogisticRegression':
                    {},
                'KNeighborsClassifier':
                    {
                        'n_neighbors': np.arange(10, 40, 3),
                        'weights': ['uniform', 'distance'],
                        #                     'leaf_size': np.arange(20, 40, 5)
                    },
                'SVC_linear':
                    {
                        'kernel': ['linear'],
                        'C': np.logspace(-7, 3, 400),
                    },
                'SVC_rbf':
                    {
                        'kernel': ['rbf', 'sigmoid'],
                        'C': np.logspace(-7, 3, 30),
                        'gamma': np.logspace(-7, 3, 30),
                    },
                'SVC_poly':
                    {
                        'kernel': ['poly'],
                        'degree': [1, 2],
                        'C': np.logspace(-7, 3, 30),
                        'gamma': np.logspace(-7, 3, 30),
                    },
                'DecisionTreeClassifier':
                    {
                        'max_depth': np.arange(5, 13, 1),
                        'min_samples_split': np.arange(2, 11, 2),
                        'min_samples_leaf': np.arange(2, 11, 2),
                        'max_features': ['auto', 'sqrt', 'log2']
                    },
                'RandomForestClassifier':
                    {
                        'n_estimators': np.arange(50, 500, 100),
                        'min_samples_split': np.arange(2, 12, 3),
                        'min_samples_leaf': np.arange(2, 12, 3),
                        'max_depth': np.arange(5, 12, 2),
                        'max_features': ['sqrt', 'log2', 'auto']
                    },
                'MLPClassifier':
                    {
                        'hidden_layer_sizes': [(2, 2), (2, 4), (2, 6), (2, 8),
                                               (4, 2), (4, 4), (4, 6), (4, 8),
                                               (6, 2), (6, 4), (6, 6), (6, 8), ],
                        'activation': ['tanh'],
                        'learning_rate_init': [0.001, 0.01, 0.1],
                        'alpha': np.logspace(-7, 2, 10),
                        'early_stopping': [True],
                        'max_iter': [1000]
                    },
                'AdaBoostClassifier':
                    {
                        'n_estimators': np.arange(50, 500, 50),
                        'learning_rate': np.linspace(0.05, 0.5, 1)
                    },
                'GaussianNB':
                    {},
                'QuadraticDiscriminantAnalysis':
                    {
                        'tol': [0.0001]
                    },
                'GradientBoostingClassifier':
                    {
                        'n_estimators': np.arange(50, 600, 50),
                        'learning_rate': [0.01, 0.1, 1],
                        'min_samples_split': np.arange(2, 11, 3),
                        'min_samples_leaf': np.arange(1, 11, 3),
                        'max_depth': np.arange(5, 12, 2),
                        'max_features': ['sqrt', 'log2', 'auto']
                    }
            }
        self.candidates, self._MODEL = _valid_candidates(candidates, 'classification')
