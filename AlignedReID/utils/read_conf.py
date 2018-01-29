#-*-coding:utf-8-*-
#=======================
#读取配置文件的程序
#=======================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ConfigParser





def get_conf_int_param(session, key):
    '''
    获取配置文件中的参数
    '''
    cf = ConfigParser.ConfigParser()
    cf.read('../conf/project_conf.conf')
    return cf.getint(session, key)


def get_conf_float_param(session, key):
    '''
    获取配置文件中的参数
    '''
    cf = ConfigParser.ConfigParser()
    cf.read('../conf/project_conf.conf')
    return cf.getfloat(session, key)


def get_conf_bool_param(session, key):
    '''
    获取配置文件中的参数
    '''
    cf = ConfigParser.ConfigParser()
    cf.read('../conf/project_conf.conf')
    return cf.getboolean(session, key)


def get_conf_str_param(session, key):
    '''
    获取配置文件中的参数
    '''
    cf = ConfigParser.ConfigParser()
    cf.read('../conf/project_conf.conf')
    return cf.get(session, key)


def set_conf_param(session, key, value):
    '''
    设置配置文件中的参数
    '''
    cf = ConfigParser.ConfigParser()
    cf.read('../conf/project_conf.conf')
    cf.add_section(session)
    cf.set(session,key,value)
    cf.write(open("../conf/project_conf.conf","w"))




