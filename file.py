import glob
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help=".path to rename dataset")
args = vars(ap.parse_args())

input_path = args["dataset"] + '/*'

d_files = glob.glob(input_path)
n = 1 
for fn in d_files:
    file_name, file_extension = os.path.splitext(fn)
    os.rename(fn,args["dataset"] + '/pic-'+ str(n) + file_extension)
    n = n + 1