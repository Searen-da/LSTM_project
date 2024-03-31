# -*- coding: utf-8 -*-
'''
@File  : date_deal.py
@Author: Stone
@Date  : 2023/5/6 9:32
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
import inspect
import datetime
import time, re
import schedule
import gc
from tools import Params_con
from db_manage import eqal_data_read
import warnings

warnings.filterwarnings("ignore")
father_path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
dataset_path = os.path.join(father_path, 'dataset')


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")  # 删除 string 字符串末尾的指定字符
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


if __name__ == '__main__':
    data_path = dataset_path + '\kpi_data_select1100_TOP40.csv'
    data_all = pd.read_csv(data_path)
    cgi_list = sorted(data_all['小区中文名'].drop_duplicates().tolist())
    print(len(cgi_list))
    mkdir(dataset_path + r'\all')
    for cgi, group in data_all.groupby('小区中文名'):
        cgi_path = os.path.join(dataset_path + '/all/' + cgi + '.csv')
        group['上行流量_M'] = group['上行流量（Kbyte）'].apply(lambda x: x / 1024)
        group['下行流量_M'] = group['下行流量（Kbyte）'].apply(lambda x: x / 1024)

        group.to_csv(cgi_path, index=False)
