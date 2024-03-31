import pandas as pd
import numpy as np
# from tqdm import tqdm
import os, sys
import inspect
import datetime
import time, re
# import schedule
import gc
import matplotlib.pyplot as plt
# from tools import Params_con
from db_manage import eqal_data_read
import warnings
from keras_LSTM import read_data

warnings.filterwarnings("ignore")
current_path = os.path.dirname(inspect.getfile(inspect.currentframe()))


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def clear_dir(path):
    if os.path.exists(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
        print("OK delete data")
    else:
        print("文件不存在 cant find dir and file")


def main_a():
    start_time = time.perf_counter()
    # param_con = Params_con()
    # print(param_con.params())
    # param = param_con.params()

    '''manipulate path'''
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, '..', 'dataset', 'all', 'A-JZ-WLMQKFQGAJ-HLW-1.csv')
    print(file_path)

    # this is for keras LSTM model
    read_data(file_path)

    # # this is for deepAR model
    # deepAR_read_data(file_path)
    

    # df = eqal_data_read(param)

    print('执行完成 in {} seconds'.format(time.perf_counter() - start_time))


if __name__ == '__main__':
    main_a()
