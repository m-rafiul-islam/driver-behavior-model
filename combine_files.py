#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 22:02:09 2022

@author: rafiul
"""

# from glob import glob 

# # filenames = [] 

# filenames = glob('/CodesRafi/results/veersion06/pe_results_version.csv')  


# glob.glob('/CodesRafi/results/veersion06/pe_results_version.csv')  

# glob.glob('*.csv')

# import os 


# # Directory to be scanned
# path = 'CodesRafi/results'
 
# # Scan the directory and get
# # an iterator of os.DirEntry objects
# # corresponding to entries in it
# # using os.scandir() method
# obj = os.scandir(path)
 
# # List all files and directories
# # in the specified path
# print("Files and Directories in '% s':" % path) 

import os 
import pandas as pd
# path = '/Users/rafiul/Library/CloudStorage/Box-Box/ISMaRT2022/server/CodesRafi/results/version06/'
path = '/Users/rafiul/Library/CloudStorage/Box-Box/ISMaRT2022/server/CodesRafi/results/version07alpha/' 
csvfiles = []
for fname in os.listdir(path):
    if fname.endswith('.csv'):
        csvfiles.append(fname)  
        
# dataframes = [pd.read_csv(path+file) for file in csvfiles] 
# dataframes[0].join(dataframes[1 :])


data = pd.read_csv(path+csvfiles[0])

for file in csvfiles:
    dfcsv = pd.read_csv(path+file) 
    data = pd.concat([data,dfcsv]) 
    
data.reset_index(drop=True,inplace=True)    
    
    
data.to_csv(path+'parameter_estimation_combined_data.csv', index_label=False)     

# data.to_csv(path+'parameter_estimation_combined_data.csv')     
df = pd.read_csv(path+'parameter_estimation_combined_data.csv') 


url = 'https://raw.githubusercontent.com/m-rafiul-islam/driver-behavior-model/main/parameter_estimation_combined_data.csv'
data2 = pd.read_csv(url) 

data2[data2['alpha']==1]['delta'].mean()  
data2[data2['alpha']==1]['a'].mean()   

data2[data2['alpha']!=1]['delta'].mean()  
data2[data2['alpha']!=1]['a'].mean()   

data_ODE = data2[data2['alpha']==1]
data_FDE = data2[data2['alpha']!=1]

# AIC_ODE = []
# for i in range(len(data_ODE)):
#     AIC = 2*3 - 2*np.log(data_ODE['objective'][i])
#     AIC_ODE.append(AIC) 





