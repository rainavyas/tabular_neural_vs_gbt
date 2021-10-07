'''
Similar to original partitioner but only generates specifically
xclim and xtime files
'''
import numpy as np
import pandas as pd
import bisect
from sklearn.utils import shuffle

class Config():
    '''
    Define Configuration for partioning data
    '''
    def __init__(self, time_splits = [0.6, 0.1, 0.15, 0.15], climate_splits = [3,1,1], in_domain_splits=[0.7, 0.15, 0.15], seed=1, eval_dev_overlap=True):
        '''
        time_splits: fractions associated with TRAIN, GAP, DEV_OUT, EVAL_OUT (split on time)
        climate_splits: number of climates kept for TRAIN, DEV_OUT, EVAL_OUT (from above time splits)
        in_domain_splits: Separation TRAIN time and specified climate segment block into TRAIN, DEV_IN, EVAL_IN
        eval_dev_overlap: Flag if TRUE, EVAL_OUT climates kept include the DEV_OUT climates kept.
        '''
        self.time_splits = time_splits
        self.climate_splits = climate_splits
        self.in_domain_splits = in_domain_splits
        self.seed = seed
        self.eval_dev_overlap = eval_dev_overlap

class XPartitioner():
    '''
    Requires a block of data and partitions it into
    the following subsets:

    1) train_xclim.csv:
                    Extra Climate Data for training.
    
    2) train_xtime.csv:
                    Extra Time Data for training.

    3) dev_xclim.csv:
                    Extra Climate Data for dev tuning.
    
    4) train_xtime.csv:
                    Extra Time Data for dev tuning.
    '''

    def __init__(self, data_path, climate_info_path, config=Config()):

        # Define the 5 climate types
        self.CLIMATES = ['tropical', 'dry', 'mild temperate', 'snow', 'polar']
        self.config = config
        # Read in the raw data
        self.df = pd.read_csv(data_path)
        # Introduce an additional column for the climate type
        self._include_climate(climate_info_path)
        # Partition the data by time segments
        self._split_by_time()
        # Partition the data by climate segments
        self._split_by_climate()
        
    
    def _include_climate(self, climate_info_path):
        '''
        Add column with climate type based on location
        '''
        # Define mapping between climate code to climate name
        letter_to_name = {
            'A' : 'tropical',
            'B' : 'dry',
            'C' : 'mild temperate',
            'D' : 'snow',
            'E' : 'polar',
            'n' : 'other'
        }

        # Load data about longitude and latitude to climate type from http://hanschen.org/koppen
        df_climate_info = pd.read_csv(climate_info_path, sep='\t')
        # Get the longitudes and latitudes at every 0.5 degrees resolution on land
        climate_longitudes = list(df_climate_info['longitude'])
        climate_latitudes = list(df_climate_info['latitude'])
        # Identify one of the five climate types using the latest climate type data provided (2010)
        climate_types = [str(typ)[0] for typ in list(df_climate_info['p2010_2010'])]
        # Load the longitudes and latitudes of all raw data
        y_lats = list(self.df['fact_latitude'])
        y_longs = list(self.df['fact_longitude'])
        # Match longitudes and latitudes to closest 0.5 degree resolution to identify climate type
        y_climates = [self._get_climate(lat, long, climate_latitudes, climate_longitudes, climate_types, count) for count, (lat, long) in enumerate(zip(y_lats, y_longs))]
        # Convert climate code names to actual names and add climate information to the raw data
        y_climates = [letter_to_name[clim] for clim in y_climates]
        self.df.insert(5, 'climate', y_climates)      

    def _get_climate(self, lat, long, climate_latitudes, climate_longitudes, climate_types, count):
        """
        Map lat, long to the closest climate_latitude and climate_longitude and then identify
        corresponding climate type.
        """
        # Find index of first occurence greater than the specific longitude
        ind_climate_long_start = bisect.bisect_left(climate_longitudes, long)
        # Find index of first occurence greater than longitude + 0.5
        ind_climate_long_end = bisect.bisect_left(climate_longitudes, long+0.5)
        # Relative (to the longitude index) index to identify the closest latitude index
        rel_ind = len(climate_latitudes[ind_climate_long_start: ind_climate_long_end]) - bisect.bisect_left(climate_latitudes[ind_climate_long_start+1: ind_climate_long_end][::-1], lat) - 1
        # The overall index of the closest latitude, longitude point
        ind = ind_climate_long_start + rel_ind

        # approx_lat, approx_long = climate_latitudes[ind], climate_longitudes[ind]
        # Find the corresponding climate type
        climate = climate_types[ind]
        return climate
    
    def _split_by_time(self):
        """
        Partition the data into the main time segments.
        """
        # Sort all data in time order
        self.df = self.df.sort_values(by=['fact_time'])
        # Find the total number of data points
        num_samples = len(self.df)
        # Use the time fractions to identify the index splits for the raw data
        first_split_ind = int(num_samples*self.config.time_splits[0])
        second_split_ind = int(num_samples*self.config.time_splits[1]) + first_split_ind
        # Identify the first time segment to be used
        self._df_first_time_all = self.df.iloc[:first_split_ind]
        # Identify the second time segment to be used
        self._df_second_time_all = self.df.iloc[second_split_ind:]

    def _split_by_climate(self):
        TRAIN_FRAC = 0.9843 # Fraction of xdata for train and rest for dev

        # Use the climate split fractions to identify the climate types partitions
        clim_first_split_ind = self.config.climate_splits[0]

        # Identify the climate types to keep for the first time segment
        first_climates_keep = self.CLIMATES[clim_first_split_ind:]
        df_first_kept_climates = self._df_first_time_all.loc[self._df_first_time_all['climate'].isin(first_climates_keep)]
        # Shuffle the first time segment data and split into train and dev
        df_first_kept_climates = shuffle(df_first_kept_climates, random_state=self.config.seed)
        df_train_xclim = df_first_kept_climates[:int(TRAIN_FRAC*len(df_first_kept_climates))]
        df_dev_xclim = df_first_kept_climates[int(TRAIN_FRAC*len(df_first_kept_climates)):]
    
        # Identify the climate types to keep for the second time segment
        second_climates_keep = self.CLIMATES[:clim_first_split_ind]
        df_second_kept_climates = self._df_second_time_all.loc[self._df_second_time_all['climate'].isin(second_climates_keep)]
        # Shuffle the second time segment data and split into train and dev
        df_second_kept_climates = shuffle(df_second_kept_climates, random_state=self.config.seed)
        df_train_xtime = df_second_kept_climates[:int(TRAIN_FRAC*len(df_second_kept_climates))]
        df_dev_xtime = df_second_kept_climates[int(TRAIN_FRAC*len(df_second_kept_climates)):]
        

        self.dfs_to_save = {}
        self.dfs_to_save['train_xclim'] = df_train_xclim
        self.dfs_to_save['dev_xclim'] = df_dev_xclim
        self.dfs_to_save['train_xtime'] = df_train_xtime
        self.dfs_to_save['dev_xtime'] = df_dev_xtime

    def save(self, save_path):
        """
        Save all relevant data split files.
        """

        # Save all files
        for name, df in self.dfs_to_save.items():
            df.to_csv(save_path+'/'+name+'.csv', index=False)
            print('Saved', name)