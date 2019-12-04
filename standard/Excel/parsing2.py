# -*-  conding:utf-8 -*-

import re
import os
import json
import xlrd
import copy


# Excel解析类
class ExcelParsing:
    def __init__(self,excel_path,sheet_name,dataList):
        '''
        excel_path: excel文件地址
        sheet_name: sheet单页名称
        dataList: 前置条件类别列表,['建筑属性','居住建筑高度分类特性','非居住建筑高度分类特性','布置方式','建筑朝向',......]
        return: dict

        '''
        self.out_dict = {}
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.dataList = dataList
        self.read_excel()


    def read_excel(self):
        
        # 打开excel文件工作簿
        workbook = xlrd.open_workbook(self.excel_path)
        # 获取指定名称sheet的单页
        worksheet = workbook.sheet_by_name(self.sheet_name)

        nrows = worksheet.nrows
        ncols = worksheet.ncols
        flag = True

        # 间距规范行列范围
        list_range = list()
        # "列名"该列位置
        condition_location = list()
        name = '列名'
        list1 = self.excel_find(nrows, ncols, flag, worksheet, name)
        condition_location = copy.deepcopy(list1)

        for i in range(1,nrows):
            data = worksheet.cell_value(i,condition_location[1])
            if isinstance(data,str)  and  '\n' in data:
                data = data.replace('\n','')
            if data in self.dataList:
                self.out_dict[data] = {}
                # 调用建筑属性
                count = 1
                while worksheet.cell_value(i+count,condition_location[1]) == '':
                    count += 1
                if data == '投影方式' or data == '重叠方式':
                    for pl in range(count):
                        self.out_dict[data][str(pl+1)] = worksheet.cell_value(i+pl,condition_location[1]+3)
                else:
                    for num in range(count):
                        res = worksheet.cell_value(i+num,condition_location[1]+2).replace('\n','')
                        # self.out_dict[data][worksheet.cell_value(i+num,condition_location[1]+2)] = worksheet.cell_value(i+num,condition_location[1]+3)
                        self.out_dict[data][res] = worksheet.cell_value(i+num,condition_location[1]+3)

    # excel遍历范围
    def excel_find(self, nrows, ncols, flag, worksheet, name):
        list1 = []
        for m in range(nrows):
            if flag:
                for n in range(ncols):
                    if worksheet.cell_value(m,n) == name:
                        list1 = [m,n]
                        flag = False
                        break
            else:
                break

        return list1

class Parsing:

    def __init__(self,out_dict):
        self.out_dict = out_dict
        self.building_attribute()
        self.live_building_category()
        self.n_live_building_category()
        self.building_towards()
        self.decorate_way()


    # 建筑属性解析
    def building_attribute(self):
        for key,value in self.out_dict['建筑属性'].items():
            value = value.split('|')
            self.out_dict['建筑属性'][key] = value
        if '' in self.out_dict['建筑属性'].keys():
            del self.out_dict['建筑属性']['']


    # 居住建筑高度分类特性解析
    def live_building_category(self):

        for key,value in self.out_dict['居住建筑高度分类特性'].items():
            value = value.replace('层数','floor').replace('建筑高度','height').replace('&','and').replace('|','or')
            self.out_dict['居住建筑高度分类特性'][key] = value
        if '' in self.out_dict['居住建筑高度分类特性'].keys():
            del self.out_dict['居住建筑高度分类特性']['']


    def n_live_building_category(self):

        for key,value in self.out_dict['非居住建筑高度分类特性'].items():
            value = value.replace('层数','floor').replace('建筑高度','height').replace('&','and').replace('|','or')
            self.out_dict['非居住建筑高度分类特性'][key] = value
        if '' in self.out_dict['非居住建筑高度分类特性'].keys():
            del self.out_dict['非居住建筑高度分类特性']['']

    # 建筑朝向解析
    def building_towards(self):
        for key,value in self.out_dict['建筑朝向'].items():
            # 数据整理
            outer_list = self.detail_parsing(key, value)
            outer_lists = copy.deepcopy(outer_list)
            count = 0
            flag = True
            # 判断条件中是否是有中文字符
            for str1 in outer_list:
                result = re.compile(u'[\u4e00-\u9fa5]')
                if result.search(str1):
                    flag = False
                    break
            if flag:
                for i in range(len(outer_list)):
                    if outer_list[i] == 'or' or outer_list[i] == 'and':
                        outer_lists.insert(i+count,' ')
                        outer_lists.insert(i+count+2,' ')
                        outer_lists.insert(i+count+3,'angle')
                        count +=3
                outer_lists.insert(0,'angle')
                outer_lists = ''.join(outer_lists)
            else:
                # 建筑朝向--以南偏东等中文字符表示
                for item in range(len(outer_list)):
                    bearing_list = {'东':'0', '北':'90', '西':'180', '南':'270'}
                    if '正' in outer_list[item]:
                        outer_list[item] = 'angle' + '==' + bearing_list[outer_list[item][1]]
                    elif '偏' in outer_list[item]:
                        now_location = outer_list[item].index('偏')
                        number = re.findall("\d+", outer_list[item])
                        if '东' and '南' in outer_list[item]:
                            bearing_list['东'] = '360'
                        if bearing_list[outer_list[item][now_location-1]] > bearing_list[outer_list[item][now_location+1]]:
                            outer_list[item] = str(int(bearing_list[outer_list[item][now_location-1]]) - int(number[0])) + '<=angle<=' + \
                                    (bearing_list[outer_list[item][now_location-1]])
                        else:
                            outer_list[item] = bearing_list[outer_list[item][now_location-1]] + '<=angle<=' + \
                                    str(int(bearing_list[outer_list[item][now_location-1]]) + int(number[0]))
                    outer_lists[item] = outer_list[item]

                for i in range(len(outer_list)):
                    if outer_list[i] == 'or' or outer_list[i] == 'and':
                        outer_lists.insert(i+count,' ')
                        outer_lists.insert(i+count+2,' ')
                        count +=2
                outer_lists = ''.join(outer_lists)
            self.out_dict['建筑朝向'][key] = outer_lists
        if '' in self.out_dict['建筑朝向'].keys():
            del self.out_dict['建筑朝向']['']

    # 布置方式
    def decorate_way(self):

        for key, value in self.out_dict['布置方式'].items():
            value = ''.join(value.split())
            if value == '':
                self.out_dict['布置方式'][key] = ''
            else:
                outer_list = self.detail_parsing(key, value)
                outer_lists = copy.deepcopy(outer_list)
                count = 0
                if key == '正向无重叠':
                    for i in range(len(outer_list)):
                        if outer_list[i] == 'or' or outer_list[i] == 'and':
                            outer_lists.insert(i+count,' ')
                            outer_lists.insert(i+count+2,' ')
                            outer_lists.insert(i+count+3,'length')
                            count +=3
                    outer_lists.insert(0,'length')
                    outer_lists = ''.join(outer_lists)
                    self.out_dict['布置方式'][key] = outer_lists
                else:
                    for i in range(len(outer_list)):
                        if outer_list[i] == 'or' or outer_list[i] == 'and':
                            outer_lists.insert(i+count,' ')
                            outer_lists.insert(i+count+2,' ')
                            outer_lists.insert(i+count+3,'inc_angle')
                            count +=3
                    outer_lists.insert(0,'inc_angle')
                    outer_lists = ''.join(outer_lists)
                    self.out_dict['布置方式'][key] = outer_lists

        if '' in self.out_dict['布置方式'].keys():
            del self.out_dict['布置方式']['']
                

    # 切割，替换
    def detail_parsing(self,key,value):
        # 数据整理
        value = value.replace(' ','').replace('&','and').replace('|','or')
        value = value.split('and')
        outer_list = []

        # 位置记录
        count = 0
        for i in range(len(value)-1):
            value.insert(i+count+1,'and')
            count += 1

        for k in value:
            count = 0
            if 'or' in k:
                inner_list = k.split('or')
                for i in range(len(inner_list)-1):
                    inner_list.insert(i+count+1,'or')
                    count += 1
                outer_list.extend(inner_list)
            else:
                outer_list.append(k)
        return outer_list
                   

class Output:
    
    def __init__(self,filePath):
        self.filePath = filePath
        self.outTojson()
    
    # 文件输出
    def outTojson(self):
        # 写成json文件
        with open(self.filePath, "w", encoding='utf-8') as output_file:
            output_file.write(json.dumps(object2.out_dict, ensure_ascii= False, indent=4))


if __name__ == '__main__':
    
    # 获取excel文件路径
    excel_path = os.path.join(os.getcwd(), 'input.xlsx')
    dataList = ['建筑属性','居住建筑高度分类特性','非居住建筑高度分类特性','投影方式','重叠方式','布置方式','建筑朝向','区域类型','建筑位置','建筑夹角','建筑相对关系','相对山墙设施','山墙设施','相对山墙宽度','山墙宽度','建筑形态','建筑面宽','山墙投影宽度','正向重叠长度','地上层数','地下层数','首层性质','首层高度','首层架空','低层非住宅高度']
    sheet_name = '规范名词注解配置列表'
    # 输出文件地址
    filePath = os.path.join(os.getcwd(), 'output.json')


    # 创建对象
    object1 = ExcelParsing(excel_path,sheet_name,dataList)
    object2 = Parsing(object1.out_dict)
    object3 = Output(filePath)

    