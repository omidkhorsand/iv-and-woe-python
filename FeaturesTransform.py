import pandas as pd
import scipy.stats as stats
import pdb
import numpy as np

class CategoricalFeature():
    """
    Create a categorical feature
    """    
    def __init__(self, df, feature, target='label'):
        self.df = df
        self.feature = feature
        self.target = target

    def __generate_bins(self):
        return self.df[self.feature].unique()

    def fit(self):
        self.bins = self.__generate_bins()
        return self
    
    def transform(self, df):
        if self.bins is None:
            raise Exception('Bins are not fit to the object')
        df_lite = df[[self.feature, self.target]] 
        print(self.feature)
        print(self.bins)      
        df_lite['bin'] = df_lite[self.feature].fillna('MISSING')
        df_lite.loc[~df_lite[self.feature].isin(self.bins), 'bin'] = 'MISSING'
        self.df_lite = df_lite[['bin', self.target]]
        return df_lite[['bin', self.target]]


class ContinuousFeature():
    def __init__(self, df, feature, target='label', bin_max=20, mono_threshold=1, bin_min_size=2500):
        """
        Create a continuous feature
        """
        self.df = df
        self.feature = feature
        # self.bin_min_size = int(len(self.df) * 0.05)
        self.bin_min_size = bin_min_size
        self.bins = None
        self.target = target
        self.df_lite = None
        self.bin_max = bin_max
        self.mono_threshold = mono_threshold

    def __generate_bins(self, bins_num):
        df = self.df[[self.feature, self.target]]
        df['bin'] = pd.qcut(df[self.feature], bins_num, duplicates='drop')
        # df['bin'] = pd.cut(df[self.feature], bins_num, duplicates='drop')
        df['bin'] =  df['bin']\
                              .apply(lambda x: x.left) \
                              .astype(float)
        bins = df['bin'].dropna().unique()
        bins.sort()
        bins[0] = -np.inf
        bins = np.insert(bins, len(bins), np.inf)
        return df, bins

    def __generate_correct_bins(self, bins_max=20):
        if self.bin_max is not None:
            bins_max = self.bin_max
        mono_threshold = self.mono_threshold
        for bins_num in range(bins_max, 1, -1):
            df,bins = self.__generate_bins(bins_num)
            df_grouped = pd.DataFrame(df.groupby('bin') \
                                      .agg({self.feature: 'count',
                                            self.target: 'sum'})) \
                                      .reset_index()
            r, p = stats.stats.spearmanr(df_grouped['bin'], df_grouped[self.target])

            print(f'bin, {bins_num}, r, {r}, bins, {bins}, bin_size, {df_grouped[self.feature].min()}, min, {self.bin_min_size  }')
            if (
                    abs(r)>= mono_threshold and                                          # check if woe for bins are monotonic
                    df_grouped[self.feature].min() > self.bin_min_size                   # check if bin size is greater than 5%
                    and not (df_grouped[self.feature] == df_grouped[self.target]).any()  # check if number of good and bad is not equal to 0
            ):
                break
        
        return bins

    def fit(self):
        """
        Calculate bin edges for a continuous feature and store in 'self.bins'
        """

        bins = self.__generate_correct_bins()
        self.bins = bins
        return self

    def transform(self, df):
        """
        Execute the binning of 'df' based on the bin edges calculated in fit() method
        Saves 'self.df_lite' which is a dataframe of columns 'bin' and 'label' 
        Everytime this method is run 'self.df_lite' changes 
        """

        if self.bins is None:
            raise Exception('Bins are not fit to the object')
        df_lite = df[[self.feature, self.target]]
        print(self.feature)
        print(self.bins)
        # pdb.set_trace()
        df_lite['bin'] = pd.cut(df_lite[self.feature], bins=self.bins, right=True).apply(lambda x: x.left).astype(float)
        df_lite['bin'] = df_lite['bin'].fillna('MISSING')
        self.df_lite = df_lite[['bin', self.target]]  
        return df_lite[['bin', self.target]]