import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,image_utils as iu
import fnmatch
# %matplotlib inline


files = glob.glob('../data/datasets/dogs-vs-cats/train/*') 

# for fn in files:
#     if 'cat' in os.path.basename(fn):
#         shutil.copy(fn, '../data/datasets/dogs-vs-cats/cat')
#     if 'dog' in os.path.basename(fn):
#         shutil.copy(fn, '../data/datasets/dogs-vs-cats/dog')

cat_files = glob.glob('../data/datasets/dogs-vs-cats/cat/*') 
dog_files = glob.glob('../data/datasets/dogs-vs-cats/dog/*') 
cat_train = np.random.choice(cat_files, size=150, replace=False) 
dog_train = np.random.choice(dog_files, size=150, replace=False) 

for fn in cat_train:
    shutil.copy(fn, '../data/datasets/dogs-vs-cats/data2/train/cat')
for fn in dog_train:
    shutil.copy(fn, '../data/datasets/dogs-vs-cats/data2/train/dog')

cat_val = np.random.choice(cat_files, size=50, replace=False) 
dog_val = np.random.choice(dog_files, size=50, replace=False) 

for fn in cat_val:
    shutil.copy(fn, '../data/datasets/dogs-vs-cats/data2/validation/cat')
for fn in dog_val:
    shutil.copy(fn, '../data/datasets/dogs-vs-cats/data2/validation/dog')

# cat_test = np.random.choice(cat_files, size=500, replace=False) 
# dog_test = np.random.choice(dog_files, size=500, replace=False) 

# for fn in cat_test:
#     shutil.copy(fn, '../data/datasets/dogs-vs-cats/test_data')
# for fn in dog_test:
#     shutil.copy(fn, '../data/datasets/dogs-vs-cats/test_data')