import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,image_utils as iu
import fnmatch
# %matplotlib inline



# d_files = glob.glob('../data/datasets/dogs-vs-cats/data3/train/drowning/*') 
# s_files = glob.glob('../data/datasets/dogs-vs-cats/data3/train/swimming/*') 
# d_train = np.random.choice(d_files, size=150, replace=False) 
# s_train = np.random.choice(s_files, size=150, replace=False) 
# for fn in d_train:
#     shutil.copy(fn, '../data/datasets/mlv/train/drowning')
# for fn in s_train:
#     shutil.copy(fn, '../data/datasets/mlv/train/swimming')

# d_files = glob.glob('../data/datasets/dogs-vs-cats/data3/validation/drowning/*') 
# s_files = glob.glob('../data/datasets/dogs-vs-cats/data3/validation/swimming/*') 
# d_val = np.random.choice(d_files, size=50, replace=False) 
# s_val = np.random.choice(s_files, size=50, replace=False) 
# for fn in d_val:
#     shutil.copy(fn, '../data/datasets/mlv/validation/drowning')
# for fn in s_val:
#     shutil.copy(fn, '../data/datasets/mlv/validation/swimming')



d_files = glob.glob('../data/datasets/mlv/train/drowning/*')
n = 1 
for fn in d_files:
    file_name, file_extension = os.path.splitext(fn)
    os.rename(fn,'../data/datasets/mlv/train/drowning/'+ str(n) + file_extension)
    n = n + 1