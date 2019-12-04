#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
    excel_terms2json.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    
    :author: fengzf
    :date created: 2019/11/5, 17:49:00
    :python version: 3.6
"""
import json
import re
import xlrd
import yaml

# 配置变量
Main_Term_Sheet_Name = 'Main_Term_Sheet_Name'
Main_Term_Sheet_Index = 'Main_Term_Sheet_Index'
First_Header = 'First_Header'
Last_Header = 'Last_Header'
Terms_Number_Header = 'Terms_Number_Header'
Re_Terms_Number = 'Re_Terms_Number'
Formula_Col = 'Formula_Col'
Not_Split = 'Not_Split'


class ExcelToJson(object):

    def __init__(self, input_path, config_, output_path=''):
        """
        :param input_path: str Excel文件路径
        :param config_: 配置
        :param output_path: 输出路径
        """
        self.input_path = input_path
        self.config = config_
        if output_path == '':
            self.output_path = input_path.rsplit('.', 1)[0] + '.json'
        else:
            self.output_path = output_path

    def read_excel(self) -> xlrd.sheet.Sheet:
        """
        读取Excel
        :return: sheet: sheet对象
        """
        workbook = xlrd.open_workbook(self.input_path)
        # sheet = workbook.sheet_by_name(self.config.get(Main_Term_Sheet_Name))
        sheet = workbook.sheet_by_index(self.config.get(Main_Term_Sheet_Index))
        return sheet

    def excel_to_dict(self, sheet, terms_range) -> list:
        """
        主条款区域转化为字典
        :return:数据列表
        """
        data = []
        # 循环行
        for i in range(terms_range[0]+1, terms_range[1]+1):

            terms_number = ''
            content = dict()
            # 循环列
            for j in range(terms_range[2], terms_range[3] + 1):
                key = self.cell_real_value(sheet, terms_range[0], j)
                value = self.cell_real_value(sheet, i, j)
                if value != '':
                    # if key in self.config.get('List_Col'):
                    #     content[key] = value.split('|')
                    # elif key == self.config.get(Terms_Number_Header):
                    #     terms_number = value
                    if key == self.config.get(Terms_Number_Header):
                        terms_number = value
                    elif key == self.config.get(Formula_Col):
                        content[key] = self.formula_parse(value)
                    elif key in self.config.get(Not_Split):
                        content[key] = [value]
                    else:
                        content[key] = value.split('|')

            # 根据条款序号确定下一步
            for term in data:
                if terms_number in term.values():
                    term['content'].append(content)
                    break
            else:
                term_dict = dict()
                term_dict[self.config.get(Terms_Number_Header)] = terms_number
                term_dict['content'] = []
                term_dict['content'].append(content)
                data.append(term_dict)
        return data

    @staticmethod
    def formula_parse(formula) -> dict:
        """
        解析公式
        :param formula:公式
        :return:公式字典
        """
        formula_dict = {'公式': '', '跳转': [], '跳转系数': '', '消防间距': '', '其他': ''}
        if 'H' in formula:
            formula_dict['公式'] = formula
        elif '消防间距' in formula:
            formula_dict['消防间距'] = formula
        elif '[' in formula:
            jump = formula.split('*')
            if len(jump) == 1:
                formula_dict['跳转系数'] = '1'
                formula_dict['跳转'] = jump[0][1: -1].split('&')
            else:
                formula_dict['跳转系数'] = jump[0] if '[' in jump[1] else jump[1]
                formula_dict['跳转'] = jump[1][1: -1].split('&') if '[' in jump[1] else jump[0][1: -1].split('&')
        else:
            formula_dict['其他'] = formula

        return formula_dict

    def save_json(self, data):
        """
        数据保存成json
        :return:
        """
        with open(self.output_path, 'w', encoding='utf-8') as json_file:
            # 确保没有不会转化为unicode符和确保缩进格式
            json_file.write(json.dumps(data, ensure_ascii=False, indent=4))

    @staticmethod
    def cell_real_value(sheet, row, col) -> str:
        """
        获取Excel合并单元格的实际值
        :param sheet:sheet对象
        :param row:行
        :param col:列
        :return:单元格值
        """
        # 获取合并单元格情况
        merged_cells = sheet.merged_cells

        for _ in merged_cells:
            # 正确获取单元格实际值
            if row in range(_[0], _[1]) and col in range(_[2], _[3]):
                return str(sheet.cell_value(_[0], _[2]))
        return str(sheet.cell_value(row, col))

    def get_terms_range(self, sheet) -> tuple:
        """
        获取主条款范围
        :return:主条款起始行、结束行、起始列、结束列
        """
        start_row, end_row, start_col, end_col = 0, 0, 0, 0
        terms_number_col = 0

        # 通过表头第一项确定起始行和起始列
        for i in range(0, sheet.ncols):
            for j in range(0, sheet.nrows):
                if self.cell_real_value(sheet, j, i) == self.config.get(First_Header):
                    start_row = j
                    start_col = i
                    break

        # 通过表头行确定结束列和规范序号列
        for x in range(start_col, sheet.ncols):
            if self.cell_real_value(sheet, start_row, x) == self.config.get(Terms_Number_Header):
                terms_number_col = x
            if self.cell_real_value(sheet, start_row, x) == self.config.get(Last_Header):
                end_col = x

        # 通过规范序号列确定结束行
        for y in range(start_row, sheet.nrows):
            # 使用正则匹配确定行对应规范
            if re.fullmatch(self.config.get(Re_Terms_Number), self.cell_real_value(sheet, y, terms_number_col)):
                end_row = y

        print('主条款范围：', (start_row, end_row, start_col, end_col))
        return start_row, end_row, start_col, end_col

    def run(self):
        sheet = self.read_excel()
        terms_range = self.get_terms_range(sheet)
        data = self.excel_to_dict(sheet, terms_range)
        self.save_json(data)


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
    trs = ExcelToJson(config.get('Workbook_Name'), config.get('Default'), config.get('Output_Path'))
    trs.run()
