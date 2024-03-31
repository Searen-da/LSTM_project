# # -*- coding: utf-8 -*-
# '''
# @File  : tools.py
# @Author: CMDI_AI
# @Date  : 2022/4/20 17:21
# '''
# import yaml
# from attrdict import AttrDict
# import argparse
# import os
# import datetime
# import inspect
# import warnings
#
# warnings.filterwarnings("ignore")
#
# father_path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
# config_path = os.path.join(father_path, 'config')
#
#
# def read_yaml(filepath):
#     with open(filepath, encoding='utf-8') as f:
#         config = yaml.safe_load(f)
#     return AttrDict(config)
#
#
# class Params_con(object):
#     def __init__(self):
#         self.parser = argparse.ArgumentParser()
#         self.get_config()
#
#     def get_config(self):
#         config_file = os.path.join(config_path, 'config.yaml')
#         config = read_yaml(config_file)
#         for conf in config.keys():
#             if conf not in self.params():
#                 self.parser.add_argument('--' + conf, type=str, default=config[conf])
#
#     def params(self):
#         param = self.parser.parse_args()
#         return param
