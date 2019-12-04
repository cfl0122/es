# -*- coding: utf-8 -*-
"""
Created on 2019‎年‎11‎月‎21‎日
此模块为规范模块串联框架
@author: hejieheng
"""
import json
from standard.match import match_rules
from standard.mian_term_manage1 import main_term_mannage
from copy import deepcopy
from standard.PreMG import PreManagement

CITY2NUM = {"佛山": "1",
            "菏泽": "2",
            "济宁": "3",
            "泰安": "4", }


def standard_frame(build_inf1, build_inf2, rule_name):
    # --------------------- 获取主规范解析的json信息 ---------------------
    city_num = CITY2NUM[rule_name] + "_"
    path = "standard/Excel/"+city_num+"main.json"

    with open(path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    # 满足的条款的数量
    count = 0
    # 满足的条款得到的间距要求值
    lst = []
    # 初始化前置规范管理模块（后续是否需要初始化还未知=。=）
    PreMG = PreManagement(True, city_num)
    # 得到前置条款中计算的值
    input_pre = PreMG.attributeJudgment(build_inf1, build_inf2)
    # --------------------- 将楼栋属性以及前置条款结果写入规范属性 ---------------------
    total_info = main_term_mannage(input_pre, build_inf1, build_inf2)
    # --------------------- 遍历条款找到规范属性适配的条款 ---------------------
    standard = match_rules(rules, total_info)
    count, lst = standard.building_rules_trave()
    # if count>1:
    #     raise ValueError('条款逻辑错误，出现多个值,')
    # elif count==0:
    #     raise ValueError('条款逻辑错误,都不符合')
    if lst != []:
        return max(lst)
    else:
        return False
