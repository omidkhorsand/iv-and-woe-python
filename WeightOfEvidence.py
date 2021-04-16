import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import FeaturesTransform as FT
from pandas.api.types import is_numeric_dtype
from AttributeRelevance import IV


class Bin():
    def __init__(self, df, target='label'):
        """
        Creates 'CategoricalFeature' and 'ContinuousFeature' objects for all columns in 'df' except 'target'
        """
    
        self.df = df
        self.target = target

    def fit(self):
        """
        Creates 'CategoricalFeature' and 'ContinuousFeature' objects for all columns in 'df' except 'target'
        Saves all feature objects in 'feats_dict'. Each feature in 'feat_dict' gets a calculated set of bin edges
        """

        self.feats_dict = {}

        for col in [c for c in self.df.columns if c != self.target]:
            if is_numeric_dtype(self.df[col]):
                self.feats_dict[col] = FT.ContinuousFeature(self.df, col, target=self.target).fit()
            else:
                self.feats_dict[col] = FT.CategoricalFeature(self.df, col, target=self.target).fit()
        
        return self
    
    def transform(self, df):
        """
        Fits 'df_lite' to all feature objects created in fit() method
        """

        for feat in self.feats_dict:
            self.feats_dict[feat].transform(df)

class WeightOfEvidence(IV):
    def __init__(self, target='label'):
        # self.df = df
        self.bin = None
        self.woe_dict = {}
        self.target = target

    def fit_transform(self, df):
        """
        Uses Bin() class to fit and transform all features in 'df'
        For each feature save weight of evidence and bin edges in 'woe_dict': {'feature': {'bin edge': 'woe value}}
        Transforms 'df' into 'df_processed' by replacing raw values with weight of evidence values 
        """

        self.bin = Bin(df, target=self.target)
        self.bin.fit()
        self.bin.transform(df)
        for feat in self.bin.feats_dict:
            woe_df = self._IV__calculate_woe(self.bin.feats_dict[feat])
            self.woe_dict[feat] = dict(zip(woe_df[feat], woe_df.woe))
        
        feature_list = [f for f in df.columns.tolist() if f!=self.target]
        print(feature_list)
        df_processed = pd.DataFrame(columns=feature_list)
        for feat in self.bin.feats_dict:
            feature = self.bin.feats_dict[feat]
            feature.df_lite['woe'] = feature.df_lite['bin'].map(self.woe_dict[feat])
            df_processed[feat] = feature.df_lite['woe']
        return df_processed[feature_list]
    
    def transform(self, df):
        """
        Returns a processed dataframe 'df_processed' based on 'self.woe_dict' calculated in self.fit() and the 
        fitted edge bins saved in self.bin.fit() (saved in self.bin.feat_dict[feat].bins)

        Re-generates 'df_lite' (self.bin.feat_dict[feat].df_lite) for the input dataframe 'df' (test set)         
        """

        feature_list = [f for f in df.columns.tolist() if f!=self.target]
        org_feature_list = [f for f in self.bin.feats_dict]
        print(feature_list)
        
        assert sorted(feature_list) == sorted(org_feature_list), "column mismatch between input df and expected columns"

        self.bin.transform(df)

        df_processed = pd.DataFrame(columns=feature_list)
        for feat in self.bin.feats_dict:
            feature = self.bin.feats_dict[feat]
            feature.df_lite['woe'] = feature.df_lite['bin'].map(self.woe_dict[feat])
            df_processed[feat] = feature.df_lite['woe']
        return df_processed[org_feature_list]