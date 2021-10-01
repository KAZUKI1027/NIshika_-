# import module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from matplotlib_venn import venn2
import missingno as msno
import math
import warnings

# Visulalize Class
#
# 2021/09/28 Kazuki Hirahara


class Feature_Confirmation:

    '''
    Feature Confirmation Modules

    '''

    # Visualize missing_values
    def Missing_plot(self, df: pd.DataFrame):
        msno.matrix(df)

    # Visualize train/test gap by ven
    def Venn_plot(self, train: pd.DataFrame, test: pd.DataFrame):

        columns = test.columns
        columns_num = len(columns)
        n_cols = 4
        n_rows = columns_num // n_cols + 1

        fig, axes = plt.subplots(figsize=(n_cols*3, n_rows*3),
                                 ncols=n_cols, nrows=n_rows)

        for col, ax in zip(columns, axes.ravel()):
            venn2(
                subsets=(set(train[col].unique()), set(test[col].unique())),
                set_labels=('Train', 'Test'),
                ax=ax
            )
            ax.set_title(col)

        fig.tight_layout()

    # Visulalize Correlation Heatmap
    def Corr_heatmap(self, train: pd.DataFrame, test: pd.DataFrame):
        fig, axs = plt.subplots(nrows=2, figsize=(15, 15))

        sns.heatmap(train.corr(
        ), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})
        sns.heatmap(test.corr(), ax=axs[1], annot=True,
                    square=True, cmap='coolwarm', annot_kws={'size': 14})

        for i in range(2):
            axs[i].tick_params(axis='x', labelsize=14)
            axs[i].tick_params(axis='y', labelsize=14)

        axs[0].set_title('Training Set Correlations', size=15)
        axs[1].set_title('Test Set Correlations', size=15)

        plt.show()

    # Visulalize Numeric Features
    def Numeric_features_plot(self, train: pd.DataFrame, test: pd.DataFrame, cont_features: list, height, figsize):

        ncols = 2
        nrows = int(math.ceil(len(cont_features)/2))

        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(height*2, height*nrows))
        plt.subplots_adjust(right=1.5, hspace=.3)

        for i, feature in enumerate(cont_features):
            plt.subplot(nrows, ncols, i+1)

            # Distribution of target features
            sns.distplot(train[feature], label='Train',
                         hist=True, color='#e74c3c')
            sns.distplot(test[feature], label='Test',
                         hist=True, color='#2ecc71')
            plt.xlabel('{}'.format(feature), size=figsize, labelpad=15)
            plt.ylabel('Density', size=figsize, labelpad=15)
            plt.tick_params(axis='x', labelsize=figsize)
            plt.tick_params(axis='y', labelsize=figsize)
            plt.legend(loc='upper right', prop={'size': figsize})
            plt.legend(loc='upper right', prop={'size': figsize})
            plt.title('Distribution of {} Feature'.format(
                feature), size=figsize, y=1.05)

        plt.show()

    # Categorical Features
    def Categorical_features_plot(self, train: pd.DataFrame, test: pd.DataFrame, cat_features: list, height, figsize):

        ncols = 2
        nrows = int(math.ceil(len(cat_features)/2))
        train["type"] = "train"
        test["type"] = "test"
        whole_df = pd.concat([train, test], axis=0).reset_index(drop=True)

        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(height*2, height*nrows))
        plt.subplots_adjust(right=1.5, hspace=.3)

        for i, feature in enumerate(cat_features):
            plt.subplot(nrows, ncols, i+1)

            # Distribution of target features
            sns.countplot(data=whole_df, x=feature, hue="type")
            plt.xlabel('{}'.format(feature), size=figsize, labelpad=15)
            plt.ylabel('Density', size=figsize, labelpad=15)
            plt.tick_params(axis='x', labelsize=figsize)
            plt.tick_params(axis='y', labelsize=figsize)
            plt.legend(loc='upper right', prop={'size': figsize})
            plt.legend(loc='upper right', prop={'size': figsize})
            plt.title('Count of {} Feature'.format(feature), size=figsize, y=1.05)

        plt.show()
