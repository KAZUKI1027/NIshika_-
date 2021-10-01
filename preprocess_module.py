import pandas as pd
import numpy as np
from sklearn import preprocessing
import category_encoders as ce

# Preprocessing Class
#
# 2021/09/28 Kazuki Hirahara


class ScalerModule:
    '''
    Scaling Modules

    '''

    # Scaling
    def StandardScaler_(self, _input_df: pd.DataFrame, target_cols: list):

        features = _input_df[target_cols]
        scaler = preprocessing.StandardScaler().fit(features.values)
        features = scaler.transform(features.values)

        _input_df[target_cols] = features

        return _input_df

    def MinMaxScaler_(self, _input_df: pd.DataFrame, target_cols: list):

        features = _input_df[target_cols]
        scaler = preprocessing.MinMaxScaler().fit(features.values)
        features = scaler.transform(features.values)

        _input_df[target_cols] = features

        return _input_df

    def Normalizer_(self, _input_df: pd.DataFrame, target_cols: list, norm=["l2", "l2", "max"]):

        features = _input_df[target_cols]
        scaler = preprocessing.Normalizer(norm=norm).fit(features.values)
        features = scaler.transform(features.values)

        _input_df[target_cols] = features

        return _input_df

    def RobustScaler_(self, _input_df: pd.DataFrame, target_cols: list):

        features = _input_df[target_cols]
        scaler = preprocessing.RobustScaler().fit(features.values)
        features = scaler.transform(features.values)

        _input_df[target_cols] = features

        return _input_df

    def FunctionTransformer_(self, _input_df: pd.DataFrame, target_cols: list, function):

        features = _input_df[target_cols]
        scaler = preprocessing.FunctionTransformer(
            function).fit(features.values)
        features = scaler.transform(features.values)

        _input_df[target_cols] = features

        return _input_df


class Encoding_Module:
    '''
    Encoding Modules

    '''

    def CountEncoding_(self, target_cols, _input_df: pd.DataFrame):

        features = _input_df[target_cols]
        encoder = ce.CountEncoder().fit(features.values)
        output_df = pd.DataFrame(encoder.transform(features.values))
        output_df.columns = target_cols
        output_df = output_df.add_prefix("CE_")

        output_df = pd.concat([_input_df, output_df], axis=1)

        return output_df

    def LabelEncoding_(self, target_cols, _input_df: pd.DataFrame):

        features = _input_df[target_cols]
        encoder = ce.OrdinalEncoder().fit(features.values)
        output_df = pd.DataFrame(encoder.transform(features.values))
        output_df.columns = target_cols
        output_df = output_df.add_prefix("LE_")

        output_df = pd.concat([_input_df, output_df], axis=1)

        return output_df

    def OneHotEncoding_(self, target_cols, _input_df: pd.DataFrame):

        features = _input_df[target_cols]
        encoder = ce.OneHotEncoder(use_cat_names=True).fit(features.values)
        output_df = pd.DataFrame(encoder.transform(features.values))
        output_df.columns = output_df.columns.str[2:]
        output_df = output_df.add_prefix("OHE_")

        output_df = pd.concat([_input_df, output_df], axis=1)

        return output_df

    def TargetEncoding_(self, target_cols, _input_df: pd.DataFrame, target):

        features = _input_df[target_cols]
        encoder = ce.TargetEncoder().fit(features.values, target)
        output_df = pd.DataFrame(encoder.transform(features.values))
        output_df.columns = target_cols
        output_df = output_df.add_prefix("TE_")

        output_df = pd.concat([_input_df, output_df], axis=1)

        return output_df


class Column_Function_Module:
    '''
    Function Modules between columns
    '''

    def ArithmeticOperation(self, _input_df, target_column1: str, target_column2: str, operation: str):
        output_df = _input_df.copy()
        output_df_columns_name = f'{target_column1}{operation}{target_column2}'

        if operation == "+":
            output_df[output_df_columns_name] = output_df[target_column1] + \
                output_df[target_column2]

        elif operation == "-":
            output_df[output_df_columns_name] = output_df[target_column1] - \
                output_df[target_column2]

        elif operation == "*":
            output_df[output_df_columns_name] = output_df[target_column1] * \
                output_df[target_column2]

        elif operation == "/":
            output_df[output_df_columns_name] = output_df[target_column1] / \
                output_df[target_column2]

        output_df = pd.concat(
            [_input_df, output_df[output_df_columns_name]], axis=1)

        return output_df

    def AggregateOperation(self, _input_df: pd.DataFrame,
                           key: str,
                           agg_column: str,
                           agg_funcs,
                           fillna=None):

        if fillna:
            _input_df[agg_column] = _input_df[agg_column].fillna(fillna)

        group_df = _input_df.groupby(key).agg(
            {agg_column: agg_funcs}).reset_index()
        column_names = [
            f'GP_{agg_column}@{key}_{agg_func}' for agg_func in agg_funcs]

        group_df.columns = [key] + column_names
        output_df = pd.merge(
            _input_df[key], group_df, on=key, how="left").drop(columns=[key])
        output_df = pd.concat(
            [_input_df, output_df], axis=1)

        return output_df
