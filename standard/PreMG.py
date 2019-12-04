# -*- coding: utf-8 -*-
"""
Created on 2019‎年‎11‎月‎6‎日，‏‎14:54:42
此模块为前置规范管理模块，用于管理前置规范，并生成出主规范管理模块所需要的参数。
@author: hejieheng
"""
import json
import os
import copy
import math
from standard.pre_cal_space import PreCalSpace
from standard.cal_overlop import cal_overlop


class PreManagement:
    def __init__(self, updata_flag, rule_name):
        # 是否需要更新json的标志位 
        self.updata_flag = updata_flag
        # 需要读取的规范的城市名
        self.rule_name = rule_name
        # self.updateConfiguration()
        self.configurationRead()
        

    def updateConfiguration(self):
        if self.updata_flag == True:
            pass

    def configurationRead(self):
        with open("standard/Excel/"+self.rule_name+"pre.json",'r',encoding='UTF-8') as load_f:
            # 前置规范读取的json数据
            self.pre_data = json.load(load_f)

    # def shadowAnalysis(self):

    def attributeJudgment(self,build_inf1,build_inf2):
        result = {}
        attribute_list = ["居住", "住宅"]
        notatrr_list = ["非", "老年人"]
        for i in range(len(list(self.pre_data["建筑属性"].keys()))):
            for attribute in attribute_list:
                if attribute in list(self.pre_data["建筑属性"].keys())[i]:
                    key_flag = True
                    for notatrr in notatrr_list:
                        if notatrr in list(self.pre_data["建筑属性"].keys())[i]:
                            key_flag = False
                            break
                    if key_flag == True:
                        key = list(self.pre_data["建筑属性"].keys())[i]

        # 建筑属性——————————————————————————————————————————————————————————————————————————
        # 暂时只提取跟居住相关的属性放入建筑属性，后续修改成放入所有符合的属性
        result["建筑属性-建筑一"] = []
        build_1_attribute = "非" + key
        for i in range(len(build_inf1["type"])):
            if build_inf1["type"][i] in self.pre_data["建筑属性"][key]:
                result["建筑属性-建筑一"].append(key)
                build_1_attribute = key

        result["建筑属性-建筑二"] = []
        build_2_attribute = "非" + key
        for i in range(len(build_inf2["type"])):
            if build_inf2["type"][i] in self.pre_data["建筑属性"][key]:
                result["建筑属性-建筑二"].append(key)
                build_2_attribute = key
        # 添加“所有类型建筑”
        for allhouse in list(self.pre_data["建筑属性"].keys()):
            if self.pre_data["建筑属性"][allhouse] == ["所有类型建筑"]:
                result["建筑属性-建筑一"].append(allhouse)
                result["建筑属性-建筑二"].append(allhouse)

        # 高度分类特性——————————————————————————————————————————————————————————————————————————
        # 建筑一
        height = build_inf1["storey_height"] * build_inf1["layers"]
        floor = build_inf1["layers"]
        if build_1_attribute == key:
            H_class = copy.deepcopy(self.pre_data["居住建筑高度分类特性"])
        else:
            H_class = copy.deepcopy(self.pre_data["非居住建筑高度分类特性"])
        result["高度分类特性-建筑一"] = []
        for i in range(len(list(H_class.values()))):
            if list(H_class.values())[i] != "":
                if eval(list(H_class.values())[i]):
                    result["高度分类特性-建筑一"].append((list(H_class.keys()))[i])
        # 建筑二
        height = build_inf2["storey_height"] * build_inf2["layers"]
        floor = build_inf2["layers"]
        if build_2_attribute == key:
            H_class = copy.deepcopy(self.pre_data["居住建筑高度分类特性"])
        else:
            H_class = copy.deepcopy(self.pre_data["非居住建筑高度分类特性"])
        result["高度分类特性-建筑二"] = []
        for i in range(len(list(H_class.values()))):
            if list(H_class.values())[i] != "":
                if eval(list(H_class.values())[i]):
                    result["高度分类特性-建筑二"].append((list(H_class.keys()))[i])

        # 建筑朝向——————————————————————————————————————————————————————————————————————————
        A_class = copy.deepcopy(self.pre_data["建筑朝向"])
        # 建筑一
        result["建筑朝向-建筑一"] = []
        angle = (build_inf1["main_direction"] + 270) % 360
        for i in range(len(list(A_class.values()))):
            if list(A_class.values())[i] != "":
                if eval(list(A_class.values())[i]):
                    result["建筑朝向-建筑一"].append((list(A_class.keys()))[(list(A_class.values())).index(list(A_class.values())[i])])
        # 建筑二
        result["建筑朝向-建筑二"] = []
        angle = (build_inf2["main_direction"] + 270) % 360
        for i in range(len(list(A_class.values()))):
            if list(A_class.values())[i] != "":
                if eval(list(A_class.values())[i]):
                    result["建筑朝向-建筑二"].append((list(A_class.keys()))[(list(A_class.values())).index(list(A_class.values())[i])])

        # 建筑位置——————————————————————————————————————————————————————————————————————————
        # 建筑一外轮廓旋转(顺时针)
        wall_b1 = copy.deepcopy(build_inf1["walls"])
        for i in range(len(wall_b1)//2):
            x = copy.deepcopy(wall_b1[2*i])
            y = copy.deepcopy(wall_b1[2*i+1])
            wall_b1[2*i] = x*math.cos(math.radians(build_inf1["main_direction"])) + y*math.sin(math.radians(build_inf1["main_direction"])) + build_inf1["coordinate"][0]
            wall_b1[2*i+1] = y*math.cos(math.radians(build_inf1["main_direction"])) - x*math.sin(math.radians(build_inf1["main_direction"])) + build_inf1["coordinate"][1]
        # 建筑二外轮廓旋转(顺时针)
        wall_b2 = copy.deepcopy(build_inf2["walls"])
        for i in range(len(wall_b2)//2):
            x = copy.deepcopy(wall_b2[2*i])
            y = copy.deepcopy(wall_b2[2*i+1])
            wall_b2[2*i] = x*math.cos(math.radians(build_inf2["main_direction"])) + y*math.sin(math.radians(build_inf2["main_direction"])) + build_inf2["coordinate"][0]
            wall_b2[2*i+1] = y*math.cos(math.radians(build_inf2["main_direction"])) - x*math.sin(math.radians(build_inf2["main_direction"])) + build_inf2["coordinate"][1]
        # 测东西向是否重叠
        # 依照设计师逻辑，除东西向重叠为东西摆放，其他可当南北摆放
        overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, 90, [0,0]) #怡珺
        if overlap == True:
            if build_inf1["coordinate"][0] > build_inf2["coordinate"][0]:
                result["建筑位置-建筑一"] = "东"
                result["建筑位置-建筑二"] = "西"
            else:
                result["建筑位置-建筑一"] = "西"
                result["建筑位置-建筑二"] = "东"
        else:
            if build_inf1["coordinate"][1] < build_inf2["coordinate"][1]:
                result["建筑位置-建筑一"] = "北"
                result["建筑位置-建筑二"] = "南"
            else:
                result["建筑位置-建筑一"] = "南"
                result["建筑位置-建筑二"] = "北"   
        
        # 区域类型————————————————————————————————————————————————————————————————————————————————————————
        region_class = copy.deepcopy(self.pre_data["区域类型"])
        region = build_inf1["area"]
        for i in range(len(list(region_class.values()))):
            if region == list(region_class.values())[i]:
                result["区域类型"] = (list(region_class.keys()))[(list(region_class.values())).index(list(region_class.values())[i])]


        # 投影方式————————————————————————————————————————————————————————————————————————————————————————

        # 布置方式————————————————————————————————————————————————————————————————————————————————————————
        # 投影方式——1：垂直投影
        # 投影方式——2：北侧楼栋法线投影
        # 投影方式——3：北侧楼栋法线+垂直投影
        # 投影方式——4：南侧楼栋法线投影
        result["布置方式"] = "无适配布置方式"
        if result["建筑位置-建筑一"] == "北":
            # 若建筑一在北侧
            overlap = False
            if self.pre_data["投影方式"]["1"] == "Y" or self.pre_data["投影方式"]["3"] == "Y": # 北侧建筑投影
                overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, -build_inf1["main_direction"], [0,0]) #怡珺
            if overlap == False and self.pre_data["投影方式"]["2"] == "Y" or self.pre_data["投影方式"]["3"] == "Y": # 垂直投影
                overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, 0, [0,0]) #怡珺
            if overlap == False and self.pre_data["投影方式"]["4"] == "Y": # 南侧建筑法线投影
                overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, -build_inf2["main_direction"], [0,0]) #怡珺
            if overlap != False:
                inc_angle = abs(build_inf1["main_direction"]-build_inf2["main_direction"])
                if self.pre_data["布置方式"]["平行"] != "":
                    if eval(self.pre_data["布置方式"]["平行"]):
                        result["布置方式"] = "平行"
                if self.pre_data["布置方式"]["垂直"] != "":
                    if eval(self.pre_data["布置方式"]["垂直"]):
                        result["布置方式"] = "垂直"
                if self.pre_data["布置方式"]["非平行非垂直"] != "":
                    if eval(self.pre_data["布置方式"]["非平行非垂直"]):
                        result["布置方式"] = "非平行非垂直" 
            else:
                result["布置方式"] = "山墙相对"
        if result["建筑位置-建筑二"] == "北":
            # 若建筑二在北侧
            overlap = False
            if self.pre_data["投影方式"]["1"] == "Y" or self.pre_data["投影方式"]["3"] == "Y": # 北侧建筑投影
                overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, -build_inf2["main_direction"], [0,0]) #怡珺
            if overlap == False and self.pre_data["投影方式"]["2"] == "Y" or self.pre_data["投影方式"]["3"] == "Y": # 垂直投影
                overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, 0, [0,0]) #怡珺
            if overlap == False and self.pre_data["投影方式"]["4"] == "Y": # 南侧建筑法线投影
                overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, -build_inf1["main_direction"], [0,0]) #怡珺
            if overlap != False:
                inc_angle = abs(build_inf1["main_direction"]-build_inf2["main_direction"])
                if self.pre_data["布置方式"]["平行"] != "":
                    if eval(self.pre_data["布置方式"]["平行"]):
                        result["布置方式"] = "平行"
                if self.pre_data["布置方式"]["垂直"] != "":
                    if eval(self.pre_data["布置方式"]["垂直"]):
                        result["布置方式"] = "垂直"
                if self.pre_data["布置方式"]["非平行非垂直"] != "":
                    if eval(self.pre_data["布置方式"]["非平行非垂直"]):
                        result["布置方式"] = "非平行非垂直"  
            else:
                result["布置方式"] = "山墙相对"
        if result["建筑位置-建筑一"] == "东" or result["建筑位置-建筑一"] == "西":
            overlap = False
            overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, -build_inf1["main_direction"], [0,0]) #怡珺
            if overlap != False:
                inc_angle = abs(build_inf1["main_direction"]-build_inf2["main_direction"])
                if self.pre_data["布置方式"]["平行"] != "":
                    if eval(self.pre_data["布置方式"]["平行"]):
                        result["布置方式"] = "平行"
                if self.pre_data["布置方式"]["垂直"] != "":
                    if eval(self.pre_data["布置方式"]["垂直"]):
                        result["布置方式"] = "垂直"
                if self.pre_data["布置方式"]["非平行非垂直"] != "":
                    if eval(self.pre_data["布置方式"]["非平行非垂直"]):
                        result["布置方式"] = "非平行非垂直" 
            else:
                result["布置方式"] = "山墙相对"
            if result["布置方式"] == "无适配布置方式":
                overlap = PreCalSpace().pre_cal_space(wall_b1, wall_b2, -build_inf2["main_direction"], [0,0]) #怡珺
                if overlap == True:
                    inc_angle = abs(build_inf1["main_direction"]-build_inf2["main_direction"])
                    if self.pre_data["布置方式"]["平行"] != "":
                        if eval(self.pre_data["布置方式"]["平行"]):
                            result["布置方式"] = "平行"
                    if self.pre_data["布置方式"]["垂直"] != "":
                        if eval(self.pre_data["布置方式"]["垂直"]):
                            result["布置方式"] = "垂直"
                    if self.pre_data["布置方式"]["非平行非垂直"] != "":
                        if eval(self.pre_data["布置方式"]["非平行非垂直"]):
                            result["布置方式"] = "非平行非垂直"  
        # 正向重叠长度————————————————————————————————————————————————————————————————————————————————————————
        result["正向重叠长度"] = 0
        if self.pre_data["布置方式"]["正向无重叠"] != '' and result["布置方式"] == "平行":
            if self.pre_data["重叠方式"]["2"] == "Y":
                length = cal_overlop(wall_b1, wall_b2, 0)
                if eval(self.pre_data["布置方式"]["正向无重叠"]):
                    result["布置方式"] = "正向无重叠"
                result["正向重叠长度"] = length
            else:
                if result["建筑位置-建筑一"] == "北":
                    if self.pre_data["重叠方式"]["1"] == "Y":
                        length = cal_overlop(wall_b1, wall_b2, -build_inf1["main_direction"])
                        if eval(self.pre_data["布置方式"]["正向无重叠"]):
                            result["布置方式"] = "正向无重叠"
                        result["正向重叠长度"] = length
                if result["建筑位置-建筑二"] == "北":
                    if self.pre_data["重叠方式"]["1"] == "Y":
                        length = cal_overlop(wall_b1, wall_b2, -build_inf2["main_direction"])
                        if eval(self.pre_data["布置方式"]["正向无重叠"]):
                            result["布置方式"] = "正向无重叠"
                        result["正向重叠长度"] = length      
        # 建筑形态————————————————————————————————————————————————————————————————————————————————————————
        # 暂时先默认都是条式建筑，后续根据实际情况进行判断
        result["建筑形态-建筑一"] = "条式建筑"
        result["建筑形态-建筑二"] = "条式建筑"
        return result

if __name__ == "__main__":
    PreMG = PreManagement(True)