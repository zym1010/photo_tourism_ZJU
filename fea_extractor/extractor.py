"""
extractor.py written by Zhongwen Xu
"""
from multiprocessing import Pool
import os
import sys

def extract(img_path):
    pass

def main(dir_name, feature, num_processes):
    pool = Pool(processes = num_processes)
    filepaths = []
    for root, dirnames, fnames in os.walk(dir_name):
        for filename in fnames:
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)
    pool.apply(extract, filepaths)


if __name__ == '__main__':
    try:
        main(sys.argv[0], sys.argv[1], int(sys.argv[2]))
    except ValueError:
        print 'Usage: ./extractor.py dir_name feature [num_processes]'
