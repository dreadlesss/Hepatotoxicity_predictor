import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy import stats
import copy
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from .supervised import *

__all__ = ['MLPlot']


        
def SignificantTest(data_list, test='mannwhitneyu', ci=0.05, feature=''):
    print('Type:', test)
    for i in range(len(data_list)):
        for j in range(i+1, len(data_list)):
            if test == 'mannwhitneyu':
                test_p = stats.mannwhitneyu(data_list[i], data_list[j], alternative="two-sided").pvalue

            elif test == 'ttest':
                if (len(data_list[i])>1) & (len(data_list[j])>1):
                    # 方差齐性检验，p值大于0.05，说明满足方差相等
                    v_test = stats.levene(data_list[i], data_list[j])
                    if v_test.pvalue < 0.05:
                        my_equal_var = True
                    else:
                        my_equal_var = False
                    test_p = stats.ttest_ind(data_list[i], data_list[j], equal_var=my_equal_var).pvalue
                else:
                    print('{}, {}: '.format(i, j), 'length of the list less than 1')

            if test_p < ci:
                print('{}, {}, {}: '.format(i, j, feature), '有差异', test_p)
            else:
                print('{}, {}, {}: '.format(i, j, feature), '无差异', test_p)
                    
                    
class MLPlot():
    '''Visualize the ML results, provide useful information for making decisions.
    
    '''

    def __init__(self, model_object=None, n_jobs=-1):
        self.model_object = model_object
        if n_jobs < 1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.regression_line = {}
        self.plot_setting_params = False

    def _color_generator(self, color_list=None):
        if color_list is None:
            color_list = ['r', 'g', 'b', 'y', 'c', 'm']
        for c in color_list:
            yield c

    def _cv_regression(self, model, x_train, y_train, train_idx, test_idx, model_name=None):
        '''Reproduce the cv results, retain the True and Predict values.
        '''
        x_cv_train = x_train[train_idx]
        x_cv_test = x_train[test_idx]
        y_cv_train = y_train[train_idx]
        y_cv_test = y_train[test_idx]
        # fit the model on the taining data
        model.fit(x_cv_train, y_cv_train)
        y_cv_test_pred = model.predict(x_cv_test)
        # save the true and predict values of the test data in cv
        cv_regression_data = np.concatenate([y_cv_test.reshape(-1, 1), y_cv_test_pred.reshape(-1, 1)], axis=1)
        cv_regression_data = pd.DataFrame(cv_regression_data, columns=['true', 'predict']).set_index(test_idx)
        if model_name:
            cv_regression_data['algorithm'] = model_name
        return cv_regression_data

    def cv_regression(self, x_train, y_train, skf, model_list, model_list_name=None):
        '''
        '''
        if not isinstance(model_list, list):
            raise Exception('model_list must be a list!')
        if model_list_name is None:
            # return the (y_true, y_pred) for cross validation
            out = Parallel(
                n_jobs=self.n_jobs, verbose=True
            )(
                delayed(self._cv_regression)(model, x_train, y_train, train_idx, test_idx)
                for train_idx, test_idx in skf.split(x_train)
                for model in model_list)
        else:
            # return the (y_true, y_pred, model_name) for cross validation
            if not isinstance(model_list, list):
                raise Exception('model_list_name must be a list or None!')
            if not len(model_list) == len(model_list_name):
                raise Exception('The length of model_list and model_list_name must be same!')
            # return the (y_true, y_pred, algorithm) for cross validation
            out = Parallel(
                n_jobs=self.n_jobs, verbose=True
            )(
                delayed(self._cv_regression)(model, x_train, y_train, train_idx, test_idx, model_name)
                for train_idx, test_idx in skf.split(x_train)
                for model, model_name in zip(model_list, model_list_name))
        self.cv_regression_data = pd.concat(out)
        return self.cv_regression_data

    def cv_regression_from_model(self, model_object=None, force=False):
        if isinstance(model_object, (RegressionModel, Model)) and force:
            self.model_object = model_object
        model_object = self.model_object

        # get the cv results
        if model_object._train_data_exist:
            x_train = model_object.x_train
            y_train = model_object.y_train
            skf = model_object.KFold
            best_df = model_object.best_df
            model_list = [model_object._MODEL[i].set_params(**j) for i, j in
                          zip(best_df['algorithm'], best_df['params'])]
            model_key_list = [i for i in best_df['algorithm']]
            res = self.cv_regression(x_train, y_train, skf, model_list, model_key_list)
        else:
            raise Exception('No training data exist!')

    def plot_setting(self, figsize=None, x_label='', y_label='', title='', subtitle=7, legend=6, axes=7, xtick=6,
                     ytick=6, figuretitle=7, params=None, **other_settings):
        # figure size
        if not figsize:
            figsize = (8 / 2.54, 8 / 2.54)
        fig = plt.figure(1, figsize=figsize, dpi=900)
        self.fig = fig
        plt.rcParams['font.sans-serif'] = 'Arial'

        # label and title
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if title:
            plt.title(title)

        # word size
        if not params:
            params = {'axes.titlesize': subtitle,  # 子图上的标题字体大小
                      'legend.fontsize': legend,  # 图例的字体大小
                      'axes.labelsize': axes,  # x、y轴上的标签的字体大小
                      'xtick.labelsize': xtick,  # x轴上的标尺的字体大小
                      'ytick.labelsize': ytick,  # y轴上的标尺的字体大小
                      'figure.titlesize': figuretitle}  # 整个画布的标题字体大小
        plt.rcParams.update(params)

        # other settings
        if other_settings:
            plt.rcParams.update(other_settings)
            self.plot_setting_params = {'figsize': figsize, 'params': params, 'x_label': x_label, 'y_label': y_label,
                                        'title': title, 'other_settings': other_settings}
        else:
            self.plot_setting_params = {'figsize': figsize, 'params': params, 'x_label': x_label, 'y_label': y_label,
                                        'title': title}

    def plot_showing(self, xlim=None, ylim=None, xticks=None, yticks=None, xticks_label=None, yticks_label=None, show=True, fname=''):
        # ticks and limitations
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        if xticks:
            plt.xticks(xticks, xticks_label)
        if yticks:
            plt.yticks(yticks, yticks_label)
        if fname:
            if self.model_object:
                plt.savefig(self.model_object.data_path / fname, format='png', bbox_inches='tight', transparent=True)
            else:
                plt.savefig(fname, format='png', bbox_inches='tight', transparent=True)
        plt.show()

    def plot_scatter(self, x, y, x_label='', y_label='', title='', loc='lower right', **scatter_params):
        plt.scatter(x, y, **scatter_params)
        # label and title
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if title:
            plt.title(title)
        if 'label' in scatter_params:
            plt.legend(loc=loc)

    def plot_line(self, x, y, x_label='', y_label='', title='', loc='lower right', **line_params):
        plt.plot(x, y, **line_params)
        # label and title
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if title:
            plt.title(title)
        if 'label' in line_params:
            plt.legend(loc=loc)

    def plot_bar(self, x, y, x_label='', y_label='', title='', loc='lower right', orient='h', **bar_params):
        if orient == 'h':
            plt.barh(x, y, **bar_params)
            plt.xticks(rotation=90)
        else:
            plt.bar(x, y, **bar_params)
        # label and title
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if title:
            plt.title(title)
        if 'label' in bar_params:
            plt.legend(loc=loc)

    def plot_heatmap(self, matrix, x_ticks=None, y_ticks=None, x_ticks_labels=None, y_ticks_labels=None, title=None,
                     xlabel=None, ylabel=None, cbar_name=None, colormap=None, bar_axes=None, shrink=None, fix_min_max=None,
                     save_path=None):
        # max and min of the value
        if isinstance(matrix, pd.DataFrame):
            column = matrix.columns.to_list()
            index = matrix.index.to_list()
            matrix = matrix.values
        else:
            column = None
            index = None

        # min and max
        matrix_max = np.max(matrix[matrix == matrix])
        matrix_min = np.min(matrix[matrix == matrix])
        if matrix_max >= 1:
            matrix_max_round = np.ceil(matrix_max)
            matrix_min_round = np.floor(matrix_min)
        elif 0.1 < matrix_max < 1:
            matrix_max_round = np.ceil(matrix_max*100) / 100
            matrix_min_round = np.floor(matrix_min*100) / 100
        else:
            pass

        # fix the color bar and plot the heat map
        if fix_min_max:
            matrix_min_round, matrix_max_round = fix_min_max
            matrix_max = matrix_max_round
        norm = Normalize(vmin=matrix_min_round, vmax=matrix_max_round)

        # start plot
        '''Note modification'''
#         fig = plt.figure(1, figsize=(3 / 2.54, 10 / 2.54), dpi=900)
        '''Note modification'''
        if not colormap:
            colormap = LinearSegmentedColormap.from_list("", ['blue','white', 'red'])
        mappable = plt.imshow(matrix, cmap=colormap, norm=norm)

        # set the axes properties
        ax = plt.gca()
        tick_size = 6
        label_size = 7
        
        # for small scale of data, list all the ticks
        if matrix.shape[1] < 40 and matrix.shape[0] < 40:
            if not x_ticks:
                x_ticks = range(matrix.shape[1])
            if not y_ticks:
                y_ticks = range(matrix.shape[0])
            if not x_ticks_labels:
                if column:
                    x_ticks_labels = column
                else:
                    x_ticks_labels = range(matrix.shape[1])
            if not y_ticks_labels:
                if index:
                    y_ticks_labels = index
                else:
                    y_ticks_labels = range(matrix.shape[0])
            ax.set_xticks(x_ticks)  
            ax.set_xticklabels(x_ticks_labels, fontsize=label_size)  
            ax.set_yticks(y_ticks)  
            ax.set_yticklabels(y_ticks_labels, fontsize=label_size)
            
        ax.tick_params(axis='x', labelsize=tick_size, pad=1, rotation=0, length=4)
        ax.tick_params(axis='y', labelsize=tick_size, pad=1, rotation=0, length=4)
        plt.title(title)

        # set the label properties
        ax.set_xlabel(xlabel, labelpad=0, fontsize=label_size)
        ax.set_ylabel(ylabel, labelpad=0, fontsize=label_size)

        # defination of colorbar
        '''Note modification'''
        position = self.fig.add_axes(bar_axes)
        '''Note modification'''
        cbar = plt.colorbar(mappable, cax=position, shrink=None)
        cbar.set_label(cbar_name, fontsize=7, rotation=90, labelpad=1)
        cbar.ax.tick_params(pad=1, labelsize=6)
        cbar.set_ticks([matrix_max_round, matrix_min_round])
        cbar.set_ticklabels([matrix_max_round, matrix_min_round])
        
        # save the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.show()
        
    def plot_cv_regression_from_model(self, model_object=None, save=False, predict=True, line_legend=False,
                                      scatter_params={}, predict_scatter_params={}, line_params={}, xlim=None,
                                      ylim=None, xticks=None, yticks=None):
        if not hasattr(self, 'cv_regression_data'):
            if isinstance(model_object, (RegressionModel, ClassificationModel, Model)):
                self.cv_regression_from_model(model_object, force=True)
            elif hasattr(self, 'model_object'):
                self.cv_regression_from_model(self.model_object, force=True)
            else:
                raise Exception('No model_object detected!')

        # plot for the cv regression
        model_object = self.model_object
        plot_data = self.cv_regression_data
        for model_key in plot_data['algorithm'].unique():
            # load settings
            if self.plot_setting_params:
                self.plot_setting(**self.plot_setting_params)
            color = self._color_generator()

            # cv scatter
            plot_datum = plot_data[plot_data['algorithm'] == model_key]
            y_pred = plot_datum['predict'].values
            y_true = plot_datum['true'].values
            self.plot_scatter(y_pred, y_true, color=color.__next__(), title=model_key, **scatter_params)

            # calc the regression line
            est = LinearRegression()
            est.fit(y_pred.reshape(len(y_pred), 1), y_true.reshape(len(y_true)))
            pred_data_min = min(plot_data['predict'])
            pred_data_max = max(plot_data['predict'])
            expand = (pred_data_max - pred_data_min) / 10
            line_x_min = pred_data_min - expand
            line_x_max = pred_data_max + expand
            line_y_min = line_x_min * est.coef_ + est.intercept_
            line_y_max = line_x_max * est.coef_ + est.intercept_
            r2 = metrics.r2_score(y_pred, y_true)
            if est.intercept_ < 0:
                regression_line = 'y={:.4f}x{:.4f}\nR²={:.4f}'.format(est.coef_[0], est.intercept_, r2)
            else:
                regression_line = 'y={:.4f}x+{:.4f}\nR²={:.4f}'.format(est.coef_[0], est.intercept_, r2)
            self.regression_line[model_key] = regression_line
            if not line_legend:
                regression_line = ''

            # regression line
            self.plot_line([line_x_min, line_x_max], [line_y_min, line_y_max], label=regression_line, **line_params)

            # predict scatter
            if predict:
                for i, tag in enumerate(model_object.predict_df_res_dict):
                    # drop duplicates
                    predict_df_res = model_object.predict_df_res_dict[tag]
                    predict_df_res = predict_df_res.T.drop_duplicates().T
                    y_pred = predict_df_res[model_key]
                    y_true = predict_df_res[model_object.predict_label_col]
                    self.plot_scatter(y_pred, y_true, color=color.__next__(), label=tag, **predict_scatter_params)

            # showing and saving
            if save:
                fname = 'cv_regression_%s.png' % model_key
            else:
                fname = None

            legend = plt.legend()
            frame = legend.get_frame()
            frame.set_alpha(0)
            frame.set_facecolor('none')
            self.plot_showing(show=True, fname=fname, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)

    def plot_feature_importance(self, estimator, x_columns, orient='h', bar_params={}, save=False, model_name=''):
        if hasattr(estimator, 'coef_'):
            x_label = 'coefficient'
            importance = estimator.coef_
            importance_label = 'coefficients of %s' % model_name
        if hasattr(estimator, 'feature_importances_'):
            x_label = 'feature importance'
            importance = estimator.feature_importances_
            importance_label = 'feature importance of %s' % model_name
        df_feature = pd.Series(importance, x_columns).sort_values(ascending=False)

        # load settings
        if self.plot_setting_params:
            self.plot_setting(**self.plot_setting_params)

        # draw bar plot of importances
        self.plot_bar(df_feature.keys().tolist(), df_feature.values.tolist(), x_label=x_label, orient=orient,
                      title=importance_label, **bar_params)

        # showing and saving
        if save:
            fname = '{}.png'.format(importance_label.replace(' ', '_'))
        else:
            fname = ''
        self.plot_showing(show=True, fname=fname)

    def plot_feature_importance_from_model(self, model_object=None, orient='h', bar_params={}, save=False):
        if isinstance(model_object, (RegressionModel, Model)):
            self.model_object = model_object
        df = self.model_object.best_df
        x_columns = model_object.df_x_columns
        model_list = df['algorithm'].tolist()
        valid_model_list = ['LinearRegression',
                            'Ridge',
                            'Lasso',
                            'ElasticNet',
                            'LogisticRegression',
                            'DecisionTreeRegressor',
                            'DecisionTreeClassifier',
                            'GradientBoostingRegressor',
                            'GradientBoostingClassifier',
                            'RandomForestRegressor',
                            'RandomForestClassifier', ]

        for model in valid_model_list:
            if model in model_list:
                pickled = df[df['algorithm'] == model]['estimator'].values[0]
                est = pickle.loads(pickled)
                self.plot_feature_importance(est, x_columns, orient, bar_params, save, model)

    def plot_permutation_importance(self, model, x_train, y_train, x_columns, random_seed, metric, x_label='None',
                                    orient='h', bar_params={}, save=False, model_name=''):
        importance = {}
        y_train_pred = model.predict(x_train)
        metric_ori = metric(y_train_pred, y_train)
        for i, column in zip(range(x_train.shape[1]), x_columns):
            x_train_permutation = x_train.copy()
            np.random.seed(random_seed)
            x_train_permutation[:, i] = np.random.permutation(x_train_permutation[:, i])
            y_permutation_predict = model.predict(x_train_permutation)
            metric_permut = metric(y_permutation_predict, y_train)
            importance[column] = metric_permut - metric_ori
        df_feature = pd.Series(importance).sort_values(ascending=False)
        title = 'permutation importance of %s' % model_name
        if not x_label:
            x_label = 'increased %' % metric.__name__

        # load settings
        if self.plot_setting_params:
            self.plot_setting(**self.plot_setting_params)

        # draw bar plot of permutation importances
        self.plot_bar(df_feature.keys().tolist(), df_feature.values.tolist(), x_label=x_label, orient=orient,
                      title=title, **bar_params)

        # showing and saving
        if save:
            fname = 'permutation_importance_%s.png' % model_name
        else:
            fname = ''
        self.plot_showing(show=True, fname=fname)

    def plot_permutation_importance_from_model(self, model_object=None, save=False, random_seed=1, algorithm='',
                                               metric=metrics.mean_squared_error, x_label='None', orient='h',
                                               bar_params={}):
        if isinstance(model_object, (RegressionModel, ClassificationModel, Model)):
            self.model_object = model_object
        model_object = self.model_object
        df = model_object.best_df
        x_train = model_object.x_train
        y_train = model_object.y_train
        x_columns = model_object.df_x_columns
        if isinstance(algorithm, str):
            if algorithm == 'all':
                algorithm = df['algorithm'].tolist()
            else:
                algorithm = [algorithm]

        # calculate the permutation importance
        for alg in algorithm:
            if alg not in df['algorithm'].tolist():
                raise Exception('Key error for the algorithm! Select one from %s' % df['algorithm'].unique().tolist())
            pickled = df[df['algorithm'] == alg]['estimator'].values[0]
            model = pickle.loads(pickled)
            self.plot_permutation_importance(model, x_train, y_train, x_columns, random_seed, metric, x_label, orient,
                                             bar_params, save, alg)

    def plot_pca(self, x1, x2, x3, scale=None, component=3, figsize=None, elev=0, azim=0, project=False,
                 save_path='./data'):
        '''This is the old PCA version.
        New version for PCA analysis is: 'plot_pca_for2' and 'plot_pca_for3'.
        '''
        if isinstance(x1, np.ndarray):
            xt = np.concatenate((x1, x2, x3), axis=0)
        elif isinstance(x1, pd.DataFrame):
            xt = pd.concat((x1, x2, x3), axis=0)
        else:
            raise Exception('The type of the data is not supported!')
        # scaling each feature
        if not scale:
            xsca1 = x1
            xsca2 = x2
            xsca3 = x3
            xscat = xt
        else:
            if scale == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif scale == 'StandardScaler':
                scaler = StandardScaler()
            scaler.fit(xt)
            xsca1 = scaler.transform(x1)
            xsca2 = scaler.transform(x2)
            xsca3 = scaler.transform(x3)
            xscat = scaler.transform(xt)

        # decomposition
        if component == 3:
            pca = PCA(3)
        elif component == 2:
            pca = PCA(2)
        pca.fit(xscat)
        xpca1 = pca.transform(xsca1)
        xpca2 = pca.transform(xsca2)
        xpca3 = pca.transform(xsca3)
        xpcat = pca.transform(xscat)
        print('explained variance:', pca.explained_variance_)
        print('explained variance ratio:', pca.explained_variance_ratio_)
        print('explained variance ratio (sum):', np.sum(pca.explained_variance_ratio_))

        #     plt.rcParams['font.sans-serif'] = 'Arial'
        if not figsize:
            figsize = (3 / 2.54, 3 / 2.54)
        fig = plt.figure(1, figsize=figsize, dpi=900)
        plt.clf()

        alhpa = 1
        line_w = 0
        s = 2
        scatter_param = {'s': s, 'alpha': alhpa, 'linewidths': line_w}
        pca_ratio = pca.explained_variance_ratio_
        tick_size = 6
        label_size = 7
        legend_size = 6
        project_flag = ''

        if component == 3:
            # rect: set the location of the pic
            ax = Axes3D(fig, rect=[0, 0.05, 0.9, 0.9], elev=elev, azim=azim, auto_add_to_figure=False)
            fig.add_axes(ax)
            plt.cla()

            disx = (np.max(xpcat[:, 0]) - np.min(xpcat[:, 0])) / 5
            disy = (np.max(xpcat[:, 1]) - np.min(xpcat[:, 1])) / 5
            disz = (np.max(xpcat[:, 2]) - np.min(xpcat[:, 2])) / 5

            ax.set_xlim(np.min(xpcat[:, 0]) - disx, np.max(xpcat[:, 0]) + disx)
            ax.set_ylim(np.min(xpcat[:, 1]) - disy, np.max(xpcat[:, 1]) + disy)
            ax.set_zlim(np.min(xpcat[:, 2]) - disz, np.max(xpcat[:, 2]) + disz)

            if project:
                project_flag = '_project'
                ax.scatter(np.full(xpca1[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca1[:, 1], xpca1[:, 2],
                           **scatter_param, c='red', label='training set')
                ax.scatter(np.full(xpca2[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca2[:, 1], xpca2[:, 2],
                           **scatter_param, c='green', label='test set 1')
                ax.scatter(np.full(xpca3[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca3[:, 1], xpca3[:, 2],
                           **scatter_param, c='blue', label='test set 2')

                ax.scatter(xpca1[:, 0], np.full(xpca1[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca1[:, 2],
                           **scatter_param, c='red')
                ax.scatter(xpca2[:, 0], np.full(xpca2[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca2[:, 2],
                           **scatter_param, c='green')
                ax.scatter(xpca3[:, 0], np.full(xpca3[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca3[:, 2],
                           **scatter_param, c='blue')

                ax.scatter(xpca1[:, 0], xpca1[:, 1], np.full(xpca1[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                           **scatter_param, c='red')
                ax.scatter(xpca2[:, 0], xpca2[:, 1], np.full(xpca2[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                           **scatter_param, c='green')
                ax.scatter(xpca3[:, 0], xpca3[:, 1], np.full(xpca3[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                           **scatter_param, c='blue')

            else:
                ax.scatter(xpca1[:, 0], xpca1[:, 1], xpca1[:, 2], **scatter_param, c='red', label='training set')
                ax.scatter(xpca2[:, 0], xpca2[:, 1], xpca2[:, 2], **scatter_param, c='green', label='test set 1')
                ax.scatter(xpca3[:, 0], xpca3[:, 1], xpca3[:, 2], **scatter_param, c='blue', label='test set 2')

            # set the ticks properties
            ax.tick_params(axis='x', labelsize=tick_size, pad=-6)
            ax.tick_params(axis='y', labelsize=tick_size, pad=-5)
            ax.tick_params(axis='z', labelsize=tick_size, pad=-3)

            # set the label properties
            ax.set_xlabel('PC1 ({:.1%})'.format(pca_ratio[0]), labelpad=-12, fontsize=label_size)
            ax.set_ylabel('PC2 ({:.1%})'.format(pca_ratio[1]), labelpad=-10, fontsize=label_size)
            ax.set_zlabel('PC3 ({:.1%})'.format(pca_ratio[2]), labelpad=-10, fontsize=label_size)

            # set the legend properties
            legend = ax.legend(loc='upper right', facecolor=None, bbox_to_anchor=(0.7, 0.2, 1, 1), fontsize=legend_size)
            frame = legend.get_frame()
            frame.set_alpha(0)
            frame.set_facecolor('none')

        if component == 2:
            plt.cla()
            plt.scatter(xpca1[:, 0], xpca1[:, 1], c='red', label='Training set', **scatter_param)
            plt.scatter(xpca2[:, 0], xpca2[:, 1], c='green', label='test set 1', **scatter_param)
            plt.scatter(xpca3[:, 0], xpca3[:, 1], c='blue', label='test set 2', **scatter_param)

            # plt.xlim(-1.1, 1.4)
            # plt.ylim(-1, 1)
            # plt.xticks(np.arange(-1, 1.4, 0.5))
            # plt.yticks(np.arange(-1, 1.4, 0.5))

            # set the ticks properties
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)

            # set the label properties
            plt.xlabel('pc1 ({:.1%})'.format(pca_ratio[0]), labelpad=0, fontsize=label_size)
            plt.ylabel('pc2 ({:.1%})'.format(pca_ratio[1]), labelpad=0, fontsize=label_size)

            # set the legend properties
            legend = plt.legend(loc='upper right', facecolor=None, bbox_to_anchor=(0.9, 0, 1, 1), fontsize=legend_size)
            frame = legend.get_frame()
            frame.set_alpha(0)
            frame.set_facecolor('none')

        # save the figure
        if save_path:
            save_path = Path(save_path) / 'PCA_{}{}.png'.format(component, project_flag)
            plt.savefig(save_path, format='png', bbox_inches='tight', transparent=True)
        plt.show()

    def plot_pca_for2(self, x1, x2, scale=None, component=3, figsize=None, elev=0, azim=0, project=False, alpha=1,
                 label1='Data set 1', label2='Data set 2', c1='red', c2='green',
                 zorder1=2, zorder2=1, save_path=False):
        # zorder这个整数越大，显示的时候越靠上
        if isinstance(x1, np.ndarray):
            xt = np.concatenate((x1, x2), axis=0)
        elif isinstance(x1, pd.DataFrame):
            xt = pd.concat((x1, x2), axis=0)
        else:
            raise Exception('The type of the data is not supported!')
        # scaling each feature
        if not scale:
            xsca1 = x1
            xsca2 = x2
            xscat = xt
        else:
            if scale == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif scale == 'StandardScaler':
                scaler = StandardScaler()
            scaler.fit(xt)
            xsca1 = scaler.transform(x1)
            xsca2 = scaler.transform(x2)
            xscat = scaler.transform(xt)

        # decomposition
        if component == 3:
            pca = PCA(3)
        elif component == 2:
            pca = PCA(2)
        pca.fit(xscat)
        xpca1 = pca.transform(xsca1)
        xpca2 = pca.transform(xsca2)
        xpcat = pca.transform(xscat)
        print('explained variance:', pca.explained_variance_)
        print('explained variance ratio:', pca.explained_variance_ratio_)
        print('explained variance ratio (sum):', np.sum(pca.explained_variance_ratio_))

        #     plt.rcParams['font.sans-serif'] = 'Arial'
        if not figsize:
            figsize = (3 / 2.54, 3 / 2.54)
        fig = plt.figure(1, figsize=figsize, dpi=900)
        plt.clf()

        alpha = alpha
        line_w = 0
        s = 0.7
        scatter_param = {'s': s, 'alpha': alpha, 'linewidths': line_w}
        pca_ratio = pca.explained_variance_ratio_
        tick_size = 6
        label_size = 7
        legend_size = 6
        project_flag = ''
        label1 = label1
        label2 = label2
        c1 = c1
        c2 = c2

        if component == 3:
            # rect: set the location of the pic
            ax = Axes3D(fig, rect=[0, 0.05, 0.9, 0.9], elev=elev, azim=azim)
            fig.add_axes(ax)
            plt.cla()

            disx = (np.max(xpcat[:, 0]) - np.min(xpcat[:, 0])) / 5
            disy = (np.max(xpcat[:, 1]) - np.min(xpcat[:, 1])) / 5
            disz = (np.max(xpcat[:, 2]) - np.min(xpcat[:, 2])) / 5

            ax.set_xlim(np.min(xpcat[:, 0]) - disx, np.max(xpcat[:, 0]) + disx)
            ax.set_ylim(np.min(xpcat[:, 1]) - disy, np.max(xpcat[:, 1]) + disy)
            ax.set_zlim(np.min(xpcat[:, 2]) - disz, np.max(xpcat[:, 2]) + disz)

            if project:
                project_flag = '_project'
                ax.scatter(np.full(xpca1[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca1[:, 1], xpca1[:, 2],
                           **scatter_param, c=c1, label=label1, zorder=zorder1)
                ax.scatter(np.full(xpca2[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca2[:, 1], xpca2[:, 2],
                           **scatter_param, c=c2, label=label2, zorder=zorder2)

                ax.scatter(xpca1[:, 0], np.full(xpca1[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca1[:, 2],
                           **scatter_param, c=c1, zorder=zorder1)
                ax.scatter(xpca2[:, 0], np.full(xpca2[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca2[:, 2],
                           **scatter_param, c=c2, zorder=zorder2)

                ax.scatter(xpca1[:, 0], xpca1[:, 1], np.full(xpca1[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                           **scatter_param, c=c1, zorder=zorder1)
                ax.scatter(xpca2[:, 0], xpca2[:, 1], np.full(xpca2[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                           **scatter_param, c=c2, zorder=zorder2)

            else:
                ax.scatter(xpca1[:, 0], xpca1[:, 1], xpca1[:, 2], **scatter_param, c=c1, label=label1, zorder=zorder1)
                ax.scatter(xpca2[:, 0], xpca2[:, 1], xpca2[:, 2], **scatter_param, c=c2, label=label2, zorder=zorder2)

            # set the ticks properties
            ax.tick_params(axis='x', labelsize=tick_size, pad=-6)
            ax.tick_params(axis='y', labelsize=tick_size, pad=-5)
            ax.tick_params(axis='z', labelsize=tick_size, pad=-3)

            # set the label properties
            ax.set_xlabel('PC1 ({:.1%})'.format(pca_ratio[0]), labelpad=-12, fontsize=label_size)
            ax.set_ylabel('PC2 ({:.1%})'.format(pca_ratio[1]), labelpad=-10, fontsize=label_size)
            ax.set_zlabel('PC3 ({:.1%})'.format(pca_ratio[2]), labelpad=-10, fontsize=label_size)

            # set the legend properties
            legend = ax.legend(loc='upper right', facecolor=None, bbox_to_anchor=(0.7, 0.2, 1, 1), fontsize=legend_size)
            frame = legend.get_frame()
            frame.set_alpha(0)
            frame.set_facecolor('none')

        if component == 2:
            plt.cla()
            plt.scatter(xpca1[:, 0], xpca1[:, 1], c=c1, label=label1, **scatter_param, zorder=zorder1)
            plt.scatter(xpca2[:, 0], xpca2[:, 1], c=c2, label=label2, **scatter_param, zorder=zorder2)

            # plt.xlim(-1.1, 1.4)
            # plt.ylim(-1, 1)
            # plt.xticks(np.arange(-1, 1.4, 0.5))
            # plt.yticks(np.arange(-1, 1.4, 0.5))

            # set the ticks properties
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)

            # set the label properties
            plt.xlabel('pc1 ({:.1%})'.format(pca_ratio[0]), labelpad=0, fontsize=label_size)
            plt.ylabel('pc2 ({:.1%})'.format(pca_ratio[1]), labelpad=0, fontsize=label_size)

            # set the legend properties
            legend = plt.legend(loc='upper right', facecolor=None, bbox_to_anchor=(0.9, 0, 1, 1), fontsize=legend_size)
            frame = legend.get_frame()
            frame.set_alpha(0)
            frame.set_facecolor('none')

        # save the figure
        if save_path:
            plt.savefig(save_path, format='png', bbox_inches='tight', transparent=True)
        plt.show()

    def plot_pca_for3(self, x1, x2, x3, scale=None, component=3, figsize=None, elev=0, azim=0, project=False,
                      alpha=0.5, save_path='',
                      label1='1', label2='2', label3='3',
                      c1='red', c2='green', c3='blue',
                      zorder1=1, zorder2=2, zorder3=3,
                      hide=0):
        if isinstance(x1, np.ndarray):
            xt = np.concatenate((x1, x2, x3), axis=0)
        elif isinstance(x1, pd.DataFrame):
            xt = pd.concat((x1, x2, x3), axis=0)
        else:
            raise Exception('The type of the data is not supported!')
        # scaling each feature
        if not scale:
            xsca1 = x1
            xsca2 = x2
            xsca3 = x3
            xscat = xt
        else:
            if scale == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif scale == 'StandardScaler':
                scaler = StandardScaler()
            scaler.fit(xt)
            xsca1 = scaler.transform(x1)
            xsca2 = scaler.transform(x2)
            xsca3 = scaler.transform(x3)
            xscat = scaler.transform(xt)

        # decomposition
        if component == 3:
            pca = PCA(3)
        elif component == 2:
            pca = PCA(2)
        pca.fit(xscat)
        xpca1 = pca.transform(xsca1)
        xpca2 = pca.transform(xsca2)
        xpca3 = pca.transform(xsca3)
        xpcat = pca.transform(xscat)
        print('explained variance:', pca.explained_variance_)
        print('explained variance ratio:', pca.explained_variance_ratio_)
        print('explained variance ratio (sum):', np.sum(pca.explained_variance_ratio_))

        #     plt.rcParams['font.sans-serif'] = 'Arial'
        if not figsize:
            figsize = (3 / 2.54, 3 / 2.54)
        fig = plt.figure(1, figsize=figsize, dpi=900)
        plt.clf()

        alpha = alpha
        line_w = 0
        s = 2
        scatter_param = {'s': s, 'alpha': alpha, 'linewidths': line_w}
        pca_ratio = pca.explained_variance_ratio_
        tick_size = 6
        label_size = 7
        legend_size = 6
        project_flag = ''

        if component == 3:
            # rect: set the location of the pic
            ax = Axes3D(fig, rect=[0, 0.05, 0.9, 0.9], elev=elev, azim=azim)
            fig.add_axes(ax)
            plt.cla()

            disx = (np.max(xpcat[:, 0]) - np.min(xpcat[:, 0])) / 5
            disy = (np.max(xpcat[:, 1]) - np.min(xpcat[:, 1])) / 5
            disz = (np.max(xpcat[:, 2]) - np.min(xpcat[:, 2])) / 5

            ax.set_xlim(np.min(xpcat[:, 0]) - disx, np.max(xpcat[:, 0]) + disx)
            ax.set_ylim(np.min(xpcat[:, 1]) - disy, np.max(xpcat[:, 1]) + disy)
            ax.set_zlim(np.min(xpcat[:, 2]) - disz, np.max(xpcat[:, 2]) + disz)

            if project:
                project_flag = '_project'
                if hide != 1:
                    ax.scatter(np.full(xpca1[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca1[:, 1], xpca1[:, 2],
                               **scatter_param, c=c1, label=label1)
                if hide != 2:
                    ax.scatter(np.full(xpca2[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca2[:, 1], xpca2[:, 2],
                               **scatter_param, c=c2, label=label2)
                if hide != 3:
                    ax.scatter(np.full(xpca3[:, 0].shape, np.min(xpcat[:, 0]) - disx), xpca3[:, 1], xpca3[:, 2],
                               **scatter_param, c=c3, label=label3)

                if hide != 1:
                    ax.scatter(xpca1[:, 0], np.full(xpca1[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca1[:, 2],
                               **scatter_param, c=c1)
                if hide != 2:
                    ax.scatter(xpca2[:, 0], np.full(xpca2[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca2[:, 2],
                               **scatter_param, c=c2)
                if hide != 3:
                    ax.scatter(xpca3[:, 0], np.full(xpca3[:, 1].shape, np.min(xpcat[:, 1]) - disy), xpca3[:, 2],
                               **scatter_param, c=c3)

                if hide != 1:
                    ax.scatter(xpca1[:, 0], xpca1[:, 1], np.full(xpca1[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                               **scatter_param, c=c1)
                if hide != 2:
                    ax.scatter(xpca2[:, 0], xpca2[:, 1], np.full(xpca2[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                               **scatter_param, c=c2)
                if hide != 3:
                    ax.scatter(xpca3[:, 0], xpca3[:, 1], np.full(xpca3[:, 2].shape, np.min(xpcat[:, 2]) - disz),
                               **scatter_param, c=c3)

            else:
                if hide != 1:
                    ax.scatter(xpca1[:, 0], xpca1[:, 1], xpca1[:, 2], **scatter_param, c=c1, label=label1)
                if hide != 2:
                    ax.scatter(xpca2[:, 0], xpca2[:, 1], xpca2[:, 2], **scatter_param, c=c2, label=label2)
                if hide != 3:
                    ax.scatter(xpca3[:, 0], xpca3[:, 1], xpca3[:, 2], **scatter_param, c=c3, label=label3)

            # set the ticks properties
            ax.tick_params(axis='x', labelsize=tick_size, pad=-6)
            ax.tick_params(axis='y', labelsize=tick_size, pad=-5)
            ax.tick_params(axis='z', labelsize=tick_size, pad=-3)

            # set the label properties
            ax.set_xlabel('PC1 ({:.1%})'.format(pca_ratio[0]), labelpad=-12, fontsize=label_size)
            ax.set_ylabel('PC2 ({:.1%})'.format(pca_ratio[1]), labelpad=-10, fontsize=label_size)
            ax.set_zlabel('PC3 ({:.1%})'.format(pca_ratio[2]), labelpad=-10, fontsize=label_size)

            # set the legend properties
            legend = ax.legend(loc='upper right', facecolor=None, bbox_to_anchor=(0.7, 0.2, 1, 1), fontsize=legend_size)
            frame = legend.get_frame()
            frame.set_alpha(0)
            frame.set_facecolor('none')

        if component == 2:
            plt.cla()
            if hide != 1:
                plt.scatter(xpca1[:, 0], xpca1[:, 1], c=c1, label=label1, **scatter_param, zorder=zorder1)
            if hide != 2:
                plt.scatter(xpca2[:, 0], xpca2[:, 1], c=c2, label=label2, **scatter_param, zorder=zorder2)
            if hide != 3:
                plt.scatter(xpca3[:, 0], xpca3[:, 1], c=c3, label=label3, **scatter_param, zorder=zorder3)

            # plt.xlim(-1.1, 1.4)
            # plt.ylim(-1, 1)
            # plt.xticks(np.arange(-1, 1.4, 0.5))
            # plt.yticks(np.arange(-1, 1.4, 0.5))

            # set the ticks properties
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)

            # set the label properties
            plt.xlabel('pc1 ({:.1%})'.format(pca_ratio[0]), labelpad=0, fontsize=label_size)
            plt.ylabel('pc2 ({:.1%})'.format(pca_ratio[1]), labelpad=0, fontsize=label_size)

            # set the legend properties
            legend = plt.legend(loc='upper right', facecolor=None, bbox_to_anchor=(0.9, 0, 1, 1), fontsize=legend_size)
            frame = legend.get_frame()
            frame.set_alpha(0)
            frame.set_facecolor('none')

        # save the figure
        if save_path:
            plt.savefig(save_path, format='png', bbox_inches='tight', transparent=True)
        plt.show()
                    
    def plot_box_violin_for_features(self, feature_data, feature_list, df_label=None, showfliers=True, adjust_params={}, dtype='box',
                                     significant_test=False, save=False):
        '''Plot the boxplot for features.

        Parameters
        ----------
        feature_data : array
            Feature array
        df_label : list
            The label/name of each dataset
        adjust_params : dict
            The parameters of plt.subplots_adjust(**adjust_params)
        dtype : str
            'Box' or 'violin' or 'both'
        significant_test : str
            One parameter in ['mannwhitneyu', 'ttest']
        save : bool / str
            Save path
        '''
        def delete_outliers(sequence):
            sequence = copy.copy(sequence)
            q1, q3 = np.percentile(sequence, [25, 75])
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            sequence = [x for x in sequence if (x <= upper_adjacent_value) & (x >= lower_adjacent_value)]
            return sequence
        
        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value
        
        def set_axis_style(ax, labels):
            ax.get_xaxis().set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_xlim(0.25, len(labels) + 0.75)

        a, b = divmod(len(feature_list), 3)
        c, d = divmod(len(feature_list), 4)
        e, f = divmod(len(feature_list), 5)
        g, h = divmod(len(feature_list), 2)
        if b == 0:
            nrows = a
            ncols = 3
        elif d == 0:
            nrows = c
            ncols = 4
        elif f == 0:
            nrows = e
            ncols = 5
        elif h == 0:
            nrows = g
            ncols = 2
        else:
            nrows = a
            ncols = 3

        figsize = (ncols*4/2.54, (nrows*4+1.5)/2.54)
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=figsize, dpi=900)
        params = {'axes.titlesize': 7,  # 子图上的标题字体大小
                  'legend.fontsize': 6,  # 图例的字体大小
                  'axes.labelsize': 7,  # x、y轴上的标签的字体大小
                  'xtick.labelsize': 6,  # x轴上的标尺的字体大小
                  'ytick.labelsize': 6,  # y轴上的标尺的字体大小
                  'figure.titlesize': 7}  # 整个画布的标题字体大小
        plt.rcParams.update(params)
        widths = False
        for i in range(nrows):
            for j in range(ncols):
                n = i*nrows + j
                if n >= len(feature_list):
                    break
                ax = axs[i][j]
                data_i = [fe[:, n] for fe in feature_data]
                
                if not df_label:
                    df_label = ['set%s'%x for x in range(len(data_i))]
                box_flag = False
                
                # violin plot
                if 'violin' in dtype:
                    # box and violin plot
                    if 'box' in dtype:
                        ax.boxplot(data_i, labels=df_label,
                                   widths=[0.2, 0.2],
                                   # 异常/溢出点
                                   flierprops={"marker":"o",
                                               "markeredgewidth":0.1,
                                               "markeredgecolor":"k",
                                               "markerfacecolor":"red",
                                               "markersize":0.9,
                                               "color":"k"},
                                   # 顶、底部横线
                                   capprops={"linewidth":0.8},
                                   # 纵向须线
                                   whiskerprops={"linewidth":0.8,},
                                   patch_artist=True,
                                   boxprops={"color":"k",
                                             "facecolor":"black",
                                             "linewidth":0.1,}
                                  )
                        box_flag = True
                    else:
                        stats = [np.percentile(temp, [25, 50, 75]) for temp in data_i]
                        quartile1, medians, quartile3 = np.array(stats).T
                #         print([(round(a, 1), round(b, 1), round(c, 1)) for a, b, c in zip(quartile1, medians, quartile3)])
                        whiskers = np.array([adjacent_values(sorted(array), q1, q3) for array, q1, q3 in zip(data_i, quartile1, quartile3)])

                        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
                        inds = np.arange(1, len(data_i) + 1)
                        ax.scatter(inds, medians, marker='o', color='orange', s=1, zorder=3)
                        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=4, zorder=2)
                        ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=0.2, zorder=2)

                    # violin plot and settings
                    if not showfliers:
                        data_i = [delete_outliers(x) for x in data_i]
                    ax.violinplot(data_i,
                                  widths=[0.5 for x in range(len(data_i))],
                                  showmeans=False,
                                  showmedians=False,
                                  showextrema=False)
                    
                # box plot
                if 'box' in dtype:
                    if not box_flag:
                        ax.boxplot(data_i, labels=df_label,
                                   widths=[0.3 for x in range(len(data_i))],
                                   flierprops={"marker":"o",
                                               "markeredgewidth":0.2,
                                               "markeredgecolor":"k",
                                               "markerfacecolor":"red",
                                               "markersize":1,
                                               "color":"g"},
                                   # 顶、底部横线
                                   capprops={"linewidth":0.8},
                                   # 纵向须线
                                   whiskerprops={"linewidth":0.8,},
                                   patch_artist=True,
                                   boxprops={"color":"k",
                                             "facecolor":"g",
                                             "linewidth":0.8,},
                                   showfliers=showfliers)

                plt.subplots_adjust(**adjust_params)
                ax.set_title(feature_list[n])
                set_axis_style(ax, df_label)
                
                # significant test
                if significant_test:
                    if significant_test in ['mannwhitneyu', 'ttest']:
                        SignificantTest(data_i, test='mannwhitneyu', ci=0.05, feature=feature_list[n])
                    else:
                        raise Exception("Select one method from ['mannwhitneyu', 'ttest']")
        if save:
            if isinstance(save, str):
                plt.savefig(save, format='png', bbox_inches='tight', transparent=True)
            else:
                plt.savefig('Box_plot_{}.png'.format('_'.join(df_label)), format='png', bbox_inches='tight', transparent=True)
        plt.show()
    
    # plot the number of samples within AD based on different k and z
    def applicability_domain_num(self, model_object, k_value, z_value, point=False, save=False):
        '''The number of test samples within the AD based on different k and z

        Parameters
        ----------
        model_object : class object
            auto_regression object

        k_value : list
            the potential value of k_value

        z_value
            the potential value of z_value
        '''
        if isinstance(model_object, (RegressionModel, Model)):
            self.model_object = model_object
        model_object = self.model_object
        # plot the compounds in domain and out of domain
        n1 = len(k_value)
        n2 = len(z_value)
        matrix_num = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                model_object.AD(knn=k_value[i], weight=z_value[j], verbose=False)
                reliable_df = model_object.predict_df_res[model_object.predict_df_res['reliable'] == 'Reliable']
                num_reliable = len(reliable_df)
                matrix_num[i, j] = num_reliable

        # plot the number of compounds within domain
        max_mat = np.max(matrix_num) + 1
        min_mat = np.min(matrix_num) - 1
        z_max_dis = (max_mat - min_mat) / 5
        z_min_dis = (max_mat - min_mat) / 1
        z_max = max_mat + z_max_dis
        z_min = min_mat - z_min_dis
        norm = Normalize(vmin=min_mat, vmax=max_mat)

        fig = plt.figure(1, figsize=(4 / 2.54, 4 / 2.54), dpi=900)
        ax = Axes3D(fig, elev=25, azim=165, auto_add_to_figure=False)
        fig.add_axes(ax)
        x, y = np.meshgrid(k_value, z_value)
        z = matrix_num.T
        model_object.AD_dict['num_reliable'] = pd.DataFrame(z, columns=k_value, index=z_value)
        mappable = ax.plot_surface(x, y, z, cmap='rainbow', alpha=1, norm=norm)
        ax.set_zlim(z_min, z_max)

        # contour on the z axis
        ax.contourf(x, y, z, zdir='z', offset=z_min, cmap='rainbow', norm=norm, alpha=1)
        if point:
            ax.contourf(point[0], point[1], [[z_min,z_min,z_min],[z_min,z_min,z_min],[z_min,z_min,z_min]], cmap='magma', zdir='z', offset=z_min, norm=norm, alpha=1)

        # defination of colorbar
        position = fig.add_axes([1.08, 0.25, 0.035, 0.35])
        cbar = plt.colorbar(mappable, cax=position, shrink=0.3, format='%d')
        cbar.set_label('number of IDs', fontsize=7, rotation=90, labelpad=1)
        cbar.ax.tick_params(pad=1, labelsize=6)
        #     cbar.set_ticks([min_mat, (min_mat+max_mat)/2, max_mat])
        #     cbar.set_ticklabels([min_mat, (min_mat+max_mat)/2, max_mat])
        cbar.update_ticks()

        # set the ticks properties
        tick_size = 6
        ax.tick_params(axis='x', labelsize=tick_size, pad=-4)
        ax.tick_params(axis='y', labelsize=tick_size, pad=-5)
        ax.tick_params(axis='z', labelsize=tick_size, pad=-3)

        # set the label properties
        lebel_size = 7
        ax.set_xlabel('k', labelpad=-12, fontsize=lebel_size)
        ax.set_ylabel('z', labelpad=-12, fontsize=lebel_size)

        # save the figure
        if save:
            fname = self.model_object.data_path / 'AD_num_reliable_compounds.png'
            plt.savefig(fname, format='png', bbox_inches='tight', transparent=True)
        plt.show()
        return cbar

    # plot the MSE for samples within AD over different k and z
    def applicability_domain_MSE(self, model_object, k_value, z_value, metric='mean_squared_error', point=False, save=False):
        '''The number of test samples within the AD based on different k and z.

        Parameters
        ----------
        model_object : model object
            Model object.

        k_value : list
            The potential value of k_value.

        z_value : list
            The potential value of z_value.
            
        save : bool
            Save the results as figure.
        '''
        if isinstance(model_object, (RegressionModel, Model)):
            self.model_object = model_object
        # plot the negtive MSE for the compounds within AD
        alg_list = model_object.predict_df_res.columns
        norm_min = -0.8
        norm_max = 0
        norm = Normalize(vmin=norm_min, vmax=norm_max)
        for alg in alg_list:
            # iter the predictive results of different algorithm
            if alg == model_object.label_col or alg == 'reliable':
                continue
            # calc the ADT based on k and z
            n1 = len(k_value)
            n2 = len(z_value)
            matrix_mse = np.zeros([n1, n2])
            for i in range(n1):
                for j in range(n2):
                    model_object.AD(knn=k_value[i], weight=z_value[j], verbose=False)
                    reliable_df = model_object.predict_df_res[model_object.predict_df_res['reliable'] == 'Reliable']
                    y_true = reliable_df[model_object.label_col].values
                    y_pred = reliable_df[alg].values
                    try:
                        res = eval('metrics.%s' % metric)(y_true, y_pred)
                    except:
                        matrix_mse[i, j] = np.nan
                    else:
                        matrix_mse[i, j] = -res
            max_mat = np.max(matrix_mse[matrix_mse == matrix_mse])
            min_mat = np.min(matrix_mse[matrix_mse == matrix_mse])

            # fill the nan
            def fill(out, row_idx, col_idx):
                if np.isnan(out[row_idx, col_idx]):
                    fill(out, row_idx, col_idx + 1)
                out[row_idx, col_idx - 1] = out[row_idx, col_idx]

            out = matrix_mse.copy()
            for row_idx in range(out.shape[0]):
                for col_idx in range(0, out.shape[1]):
                    if np.isnan(out[row_idx, col_idx]):
                        fill(out, row_idx, col_idx + 1)
            matrix_mse = out.copy()

            # plot the surface
            z_max_dis = (max_mat - min_mat) / 5
            z_min_dis = (max_mat - min_mat) / 1
            z_max = max_mat + z_max_dis
            z_min = -1.2

            # set figure
            plt.rcParams['font.sans-serif'] = 'Arial'
            fig = plt.figure(1, figsize=(4 / 2.54, 4 / 2.54), dpi=900)
            ax = Axes3D(fig, elev=25, azim=165, auto_add_to_figure=False)
#             ax = Axes3D(fig, elev=25, azim=165)
            fig.add_axes(ax)
            plt.title(alg)
            x, y = np.meshgrid(k_value, z_value)
            z = matrix_mse.T
            model_object.AD_dict[alg] = pd.DataFrame(z, columns=k_value, index=z_value)
            mappable = ax.plot_surface(x, y, z, cmap='rainbow', alpha=1, norm=norm)
            ax.set_zlim(z_min, norm_max)

            # contour on the z axis
            ax.contourf(x, y, z, zdir='z', offset=z_min, cmap='rainbow', norm=norm, alpha=1)
            if point:
                ax.contourf(point[0], point[1], [[z_min,z_min,z_min],[z_min,z_min,z_min],[z_min,z_min,z_min]], cmap='magma', zdir='z', offset=z_min, norm=norm, alpha=1)

            # defination of colorbar
            position = fig.add_axes([1.08, 0.25, 0.035, 0.35])
            cbar = plt.colorbar(mappable, cax=position, shrink=0.3)
            cbar.set_label('-MSE', fontsize=7, rotation=90, labelpad=1)
            cbar.ax.tick_params(pad=1, labelsize=6)
            cbar.set_ticks([norm_max, (norm_max + norm_min) / 2, norm_min])
            cbar.set_ticklabels([norm_max, (norm_max + norm_min) / 2, norm_min])

            # set the ticks properties
            tick_size = 6
            ax.tick_params(axis='x', labelsize=tick_size, pad=-5)
            ax.tick_params(axis='y', labelsize=tick_size, pad=-4)
            ax.tick_params(axis='z', labelsize=tick_size, pad=-3)

            # set the z ticks format
            zmajorLocator = MultipleLocator(0.3)
            ax.zaxis.set_major_locator(zmajorLocator)
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # set the label properties
            lebel_size = 7
            ax.set_xlabel('k', labelpad=-12, fontsize=lebel_size)
            ax.set_ylabel('z', labelpad=-10, fontsize=lebel_size)
            # ax.set_zlabel('-MSE', labelpad=10, fontsize=lebel_size, rotation=90)

            # set the legend properties
            # legend = ax.legend(loc='upper right', facecolor=None, bbox_to_anchor=(0.7, 0.2, 1, 1), fontsize=6)
            if save:
                fname = self.model_object.data_path / 'AD_plot_{}.png'.format(alg)
                plt.savefig(fname, format='png', bbox_inches='tight', transparent=True)
            plt.show()

    def model_MSE_for_data(self, model_object=None, knn=7, weight=1.5):
        '''iter the algorithm, calculate the MSE for samples within and outside the AD.'''
        if isinstance(model_object, (RegressionModel, Model)):
            self.model_object = model_object
        model_object = self.model_object
        count = model_object.AD(knn=knn, weight=weight, verbose=True)
        alg_list = model_object.predict_df_res.columns
        alg_calc = []
        tar_calc = ['Reliable', 'Unreliable']
        for alg in alg_list:
            # iter the predictive results of different algorithm
            if alg == model_object.label_col or alg == 'reliable':
                continue
            else:
                alg_calc.append(alg)

        # calc MSE for each model
        matrix_mse = []
        for alg in alg_calc:
            alg_res = []
            for tar in tar_calc:
                target_df = model_object.predict_df_res[model_object.predict_df_res['reliable'] == tar]
                y_true = target_df[model_object.label_col].values
                y_pred = target_df[alg].values
                MSE = metrics.mean_squared_error(y_true, y_pred)
                alg_res.append(MSE)
            matrix_mse.append(alg_res)
        tar_calc = [tar_calc[0] + '(%s)' % count[1], tar_calc[1] + '(%s)' % count[0]]
        return pd.DataFrame(matrix_mse, columns=tar_calc, index=alg_calc)

    
    def plot_3D_tuning_results_for_feature(self, tuning_res, param1, param2, results, feature_col='feature_param', param1_name='x',
                                    param2_name='y',res_name='rank', title='Tuning results', save_path='3D_surface_tuning.png',
                                    elev=25, azim=165, fix_min_max=None, point=False, ymajorLocator=0.001, xmajorLocator=0.001):
        '''Plot the tuning-result surface for feature parameters, param1 as x, param2 as y, results as z.

        Parameters
        ----------
        tuning_res : DataFrame
            Model tuning results from the 'Model' object.
        param1 : str
            One of the tuning parameter.
        param2 : str
            The other tuning parameter.
        results : str
            Results of the corresponding parameters.
        feature_col ： str
            Column of the features.
        param1_name : str
            Name of the param1. It will be displayed in x axis.
        param2_name : str
            Name of the param2. It will be displayed in y axis.
        res_name : str
            Name of the results. It will be displayed in z axis.
        title : str
            Title of the plot.
        save_path : str
            Storage path.
        '''
        tuning_res.reset_index(drop=True, inplace=True)
        for i, row in tuning_res.iterrows():
            feature_param = eval(row[feature_col])
            if 'rdk_fp' in feature_param:
                for params in eval(row['feature_param'])['rdk_fp'].values():
                    tuning_res.loc[i, param1] = params[param1]
                    tuning_res.loc[i, param2] = params[param2]
            else:
                params = eval(row[feature_col])
                tuning_res.loc[i, param1] = params[param1]
                tuning_res.loc[i, param2] = params[param2]
        return self.plot_3D_tuning_results_for_model(tuning_res, param1, param2, results, param1_name, param2_name,res_name, title, save_path,
                                         elev, azim, fix_min_max, point, ymajorLocator, xmajorLocator)

    def plot_3D_tuning_results_for_model(self, tuning_res, param1, param2, results, param1_name='x', param2_name='y',res_name='rank', 
                                         title='Tuning results',save_path='3D_surface_tuning.png', elev=25, azim=165, fix_min_max=None,
                                         point=False, ymajorLocator=0.001, xmajorLocator=0.001):
        '''Plot the tuning-result surface for model parameters, param1 as x, param2 as y, results as z.

        Parameters
        ----------
        Same as function <plot_3D_tuning_results_for_feature>.
        '''
        # generate the tuning matrix
        matrix = defaultdict(dict)
        for i, row in tuning_res.iterrows():
            matrix[row[param1]][row[param2]] = row[results] 
        matrix = dict(matrix)

        # sort index
        df = pd.DataFrame(matrix).sort_index(axis=0).sort_index(axis=1)
    #     df = df.replace(np.nan, 0)
        matrix = df.values


        # max and min of the value
        matrix_max = np.max(matrix[matrix == matrix])
        matrix_min = np.min(matrix[matrix == matrix])
        if matrix_max > 1:
            matrix_max_round = np.ceil(matrix_max)
            matrix_min_round = np.floor(matrix_min)
        elif 0.1 < matrix_max < 1:
            matrix_max_round = np.ceil(matrix_max*100) / 100
            matrix_min_round = np.floor(matrix_min*100) / 100
        else:
            pass

        # control the distance of z axis
        z_max_dis = (matrix_max - matrix_min) / 5
        z_min_dis = (matrix_max - matrix_min) / 0.7
        z_max = matrix_max + z_max_dis
        z_min = matrix_min - z_min_dis

        # set figure
        plt.rcParams['font.sans-serif'] = 'Arial'
        fig = plt.figure(1, figsize=(4 / 2.54, 4 / 2.54), dpi=900)
        # ax = Axes3D(fig, elev=25, azim=165, auto_add_to_figure=False)
        ax = Axes3D(fig, elev=elev, azim=azim)
        fig.add_axes(ax)
        plt.title(title)

        # set the x, y and z
        x, y = np.meshgrid(df.columns, df.index)
        z = matrix

        # fix the color bar and plot the surface
        if fix_min_max:
            matrix_min_round, matrix_max_round = fix_min_max
            z_min_dis = (matrix_max_round - matrix_min_round) / 0.7
            z_min = matrix_min_round - z_min_dis
            matrix_max = matrix_max_round
        norm = Normalize(vmin=matrix_min_round, vmax=matrix_max_round)
        mappable = ax.plot_surface(x, y, z, cmap='rainbow', alpha=1, norm=norm)
        ax.set_zlim(z_min, matrix_max)


        # contour on the z axis
        ax.contourf(x, y, z, zdir='z', offset=z_min, cmap='rainbow', norm=norm, alpha=1)

        if point:
            ax.contourf(point[0], point[1], [[z_min,z_min,z_min],[z_min,z_min,z_min],[z_min,z_min,z_min]], cmap='magma', zdir='z', offset=z_min, norm=norm, alpha=1)

        # defination of colorbar
        position = fig.add_axes([1.08, 0.25, 0.035, 0.35])
        cbar = plt.colorbar(mappable, cax=position, shrink=0.3)
        cbar.set_label(res_name, fontsize=7, rotation=90, labelpad=1)
        cbar.ax.tick_params(pad=1, labelsize=6)
        cbar.set_ticks([matrix_max_round, matrix_min_round])
        cbar.set_ticklabels([matrix_max_round, matrix_min_round])

        # set the ticks properties
        tick_size = 6
        ax.tick_params(axis='x', labelsize=tick_size, pad=-3)
        ax.tick_params(axis='y', labelsize=tick_size, pad=-5)
        ax.tick_params(axis='z', labelsize=tick_size, pad=-3)

    #     set the y ticks format
        ymajorLocator = MultipleLocator(ymajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        xmajorLocator = MultipleLocator(xmajorLocator)
        ax.xaxis.set_major_locator(xmajorLocator)
    #     ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # set the label properties
        label_size = 7
        ax.set_xlabel(param1_name, labelpad=-10, fontsize=label_size)
        ax.set_ylabel(param2_name, labelpad=-10, fontsize=label_size)


        # save the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.show()

    def plot_2D_fingerprints_param(self, df, min_value=None, max_value=None, save_path=None, fig_title=None,
                                   cbar_label=None, xmajorLocator=1, ymajorLocator=1, line=[0.67, 0.69, 0.71],
                                   param1_name=None, param2_name=None, levels=None):
        # max and min of the value
        matrix = df.values
        matrix_max = np.max(matrix[matrix == matrix])
        matrix_min = np.min(matrix[matrix == matrix])
        if matrix_max > 1:
            matrix_max_round = np.ceil(matrix_max)
            matrix_min_round = np.floor(matrix_min)
        elif 0.1 < matrix_max < 1:
            matrix_max_round = np.ceil(matrix_max*100) / 100
            matrix_min_round = np.floor(matrix_min*100) / 100
        else:
            pass

        # draw
        if min_value:
            matrix_min_round = min_value
            matrix_max_round = max_value
        if not levels:
            levels = np.linspace(matrix_min_round, matrix_max_round, 50)

        x, y = np.meshgrid(df.columns, df.index)
        z = matrix

        norm = Normalize(vmin=matrix_min_round ,vmax=matrix_max_round)
        fig = plt.figure(1, figsize=(4.5 / 2.54, 4 / 2.54), dpi=900)
        cset = plt.contourf(x, y, z, alpha=1, norm=norm, levels = levels)
        contour = plt.contour(x, y, z, line, colors='k', linewidths=0.5, norm=norm)
        # lines in the graph
        plt.clabel(contour, fontsize=5, colors='k', fmt='%.2f')

        # set the x ticks format
        ax = plt.gca()
        ax.invert_yaxis()
        xmajorLocator = MultipleLocator(xmajorLocator)
        ax.xaxis.set_major_locator(xmajorLocator)

        # set the y ticks format
        ymajorLocator = MultipleLocator(ymajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)

        # set title
        plt.title(fig_title, fontsize=7)

        # set the ticks properties
        tick_size = 6
        ax.tick_params(axis='x', labelsize=tick_size, pad=2)
        ax.tick_params(axis='y', labelsize=tick_size, pad=2)

        # set the label properties
        label_size = 7
        ax.set_xlabel(param1_name, labelpad=2, fontsize=label_size)
        ax.set_ylabel(param2_name, labelpad=2, fontsize=label_size)

        # defination of colorbar
        # cbar = plt.colorbar(cset)
        # position = fig.add_axes([0.95, 0.125, 0.035, 0.75])
        cbar = plt.colorbar(cset)
        cbar.set_label(cbar_label, fontsize=7, rotation=90, labelpad=2)
        cbar.ax.tick_params(pad=1, labelsize=6)
        cbar.set_ticks([matrix_max_round, (matrix_max_round+matrix_min_round)/2, matrix_min_round])
        cbar.set_ticklabels([matrix_max_round, '%.2f'%((matrix_max_round+matrix_min_round)/2), matrix_min_round])        

        # plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.show()