# -*- coding: utf-8 -*-
'''
@File  : pg_client.py
@Author: CMDI_AI
@Date  : 2022/4/17 17:21
'''

import json
# import psycopg2
# from DBUtils.PooledDB import PooledDB
import math


class PGClient(object):
    __pool = None

    def __init__(self, param):
        print(param.DB_HOST)
        self.__pool = PooledDB(creator=psycopg2, mincached=param.DB_MIN_CACHED,
                               maxcached=param.DB_MAX_CACHED,
                               maxshared=param.DB_MAX_SHARED,
                               maxconnections=param.DB_MAX_CONNECYIONS,
                               blocking=param.DB_BLOCKING, maxusage=param.DB_MAX_USAGE,
                               host=param.DB_HOST, port=param.DB_PORT,
                               user=param.DB_USER, password=param.DB_PASSWORD,
                               database=param.DB_DBNAME)
        self._conn = None
        self._cursor = None
        self.__get_conn()
        print("Opened database successfully")

    def __get_conn(self):
        self._conn = self.__pool.connection()
        self._cursor = self._conn.cursor()

    def close(self):
        try:
            self._cursor.close()
            self._conn.close()
        except Exception as e:
            print(e)

    def __execute(self, sql, param=()):
        count = self._cursor.execute(sql, param)
        return count

    """查询单个结果"""

    def select_one(self, sql, param=()):
        count = self.__execute(sql, param)
        result = self._cursor.fetchone()
        self.close()
        return count, result

    """查询多个结果"""

    def select_many(self, sql, param=()):
        count = self.__execute(sql, param)
        result = self._cursor.fetchall()
        self.close()
        return count, result

    def execute(self, sql, param=()):
        count = self.__execute(sql, param)
        # print("yy")
        return count

    """
    sql示例："insert into table(username,password,userid) values(%s,%s,%s)"
    data:数据列表
    """

    def banch_insert(self, sql, data, size):
        try:
            cycles = math.ceil(len(data) / size)
            for i in range(cycles):
                val = data[i * size: (i + 1) * size]
                self._cursor.executemany(sql, val)
                self._conn.commit()
        except Exception as e:
            print(e)
            self._conn.rollback()
        # finally:
        #     self.close()

    # def banch_sql_concat(data,table_name, size):

    # 执行增删改sql
    def ExceNonQuery(self, sql, param=()):
        try:
            self._cursor.execute(sql, param)
            self._conn.commit()
        except Exception as e:
            print(e)
            self._conn.rollback()
        # finally:
        #     self.close()

    # 执行查询sql
    def ExecQuery(self, sql, param=()):
        try:
            self._cursor.execute(sql, param)
            res = self._cursor.fetchall()

        except Exception as err:
            print("查询失败, %s" % err)
        else:
            return res

    def ExceNonQueryWithSelectAlarm(self, sql_insert, sql_select, param=()):
        try:
            self._cursor.execute(sql_select, param)
            res = self._cursor.fetchall()
        except Exception as err:
            print("查询失败, %s" % err)
        else:
            self._cursor.executemany(sql_insert, res)
            self._conn.commit()

    def __del__(self):
        class_name = self.__class__.__name__
        print("{}销毁".format(class_name))
