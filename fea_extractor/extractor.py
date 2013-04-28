"""
extractor.py written by Zhongwen Xu
"""
from multiprocessing import Pool
import os

def extract(img_path):
    pass

def main(dir_name, feature):
    pool = Pool(processes = 10)
    filepaths = []
    for root, dirnames, fnames in os.walk(dir_name):
        for filename in fnames:
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)
    pool.apply(extract, filepaths)


if __name__ == '__main__':
    try:
        main(sys.argv[0], sys.argv[1])
    except ValueError:
        print 'Usage: ./extractor.py dir_name feature'
