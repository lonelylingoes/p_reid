#-*-conding:utf-8-*-
#=======================
#读取配置文件的程序
#=======================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ConfigParser


def get_conf_param(blob, sub_blob):
	'''
	获取配置文件中的参数
	input:
		blob:根节点配置项
		sub_blob:子节点配置项
	output:
		具体参数
	'''
	cf = ConfigParser.ConfigParser()
	cf.read('../project_conf.conf')
	return cf.get(blob, sub_blob)