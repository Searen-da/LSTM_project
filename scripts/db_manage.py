# -*- coding: utf-8 -*-
'''
@File  : pg_client.py
@Author: CMDI_AI
@Date  : 2022/4/17 17:21
'''

from pg_client import PGClient
# from psycopg2 import extras
# from tools import Params_con
import pandas as pd
from pandas.tseries.offsets import Hour, Minute, Day
import numpy as np
# from tqdm import tqdm
import os
import gc
import inspect

father_path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
config_path = os.path.join(father_path, 'config')


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


# 可参考这个数据类型读取方法，读取配置信息
def eqal_data_read(param):
    # try:
    #     pg = PGClient(param)
    #     sql_getdata = "select abnormal_flow_class_id, abnormal_flow_class from {}".format(param.flowclass_table_name)
    #     print(sql_getdata)
    #     res = pg.ExecQuery(sql_getdata)
    #     pg.close()
    #     del pg
    #     df_eqa = pd.DataFrame(res, columns=['abnormal_flow_class_id', 'abnormal_flow_class'])
    #     # 修改对应字段标题名
    #     # df_eqa.rename()
    # except:
    df = pd.read_csv(config_path + '\city_num.csv', encoding='utf-8')
    df_eqa = df

    print('获取到流量异常类型配置表')
    return df_eqa
