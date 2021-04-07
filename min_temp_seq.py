import numpy as np
import os, time
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='walkthrough folder containing .npy and get min tem sequence')
    parser.add_argument('--input', type=str, help="path to .npy files")
    parser.add_argument('--init_len', type = int, help = "initial temporal len")

    return parser.parse_args()


def walk_through_files(path, file_extension='.npy'):
   for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
         if filename.endswith(file_extension): 
            yield os.path.join(dirpath, filename)
#--------------------------------------------------------


def get_min_sequence(min_len,path):
    start = datetime.now()
    
    #data_folder = os.path.join(folder, 'DATA')
    #list_npy = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    
    min_len = min_len
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npy") and 'QA' not in file:
                 file_path = os.path.join(root,file)

                 try:

                     data = np.load(file_path)
                     data_shape = data.shape[0]
                     count +=1 
                     if data_shape < min_len:
                        min_len = data_shape
                     print(count)
                 except:
                     print('error in file ', file)
    
    
    print('total files traversed ', count)
    print('minimum temporal length ', min_len)
    print('total elapsed time ', datetime.now() - start)



if __name__ == '__main__':

# # --------------------------------------------------------------------------
    args = parse_args()
    path = args.input
    min_len = args.init_len
    get_min_sequence(min_len,path)
