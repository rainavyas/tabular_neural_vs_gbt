'''
Generates the following datasets:

- train_xclim.csv
- dev_xclim.csv
- train_xtime.csv
- dev_xtime.csv

Refer to the diagram to see what this means with regard to time and climate
'''

import argparse
import os
import sys
from x_partitioner import XPartitioner, Config

if __name__ == '__main__':

    commandLineParser = argparse.ArgumentParser(description='Partition data.')
    commandLineParser.add_argument('data_path', type=str, help='Path to data')
    commandLineParser.add_argument('climate_info_path', type=str, help='Path to climate information')
    commandLineParser.add_argument('--time_splits', nargs=4, type=float, default=[0.6, 0.1, 0.15, 0.15], help='Time splits')
    commandLineParser.add_argument('--climate_splits', nargs=3, type=int, default=[3, 1, 1], help='Climate splits')

    args = commandLineParser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/generate_x.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the configurable parameters
    config = Config(args.time_splits, args.climate_splits)

    # Partition the data
    partitioner = XPartitioner(args.data_path, args.climate_info_path, config)
    # Print number of data points in each data split
    for name, df in partitioner.dfs_to_save.items():
        print(name, df.shape[0])
    
    # Save all files
    partitioner.save(args.save_path)
