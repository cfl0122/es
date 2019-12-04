import math


class match_rules():
    def __init__(self,full_rules,all_info):
        self.full_rules=full_rules
        self.rules={}
        self.all_info=all_info
        self.ban_rule=''

    # --------------------- 条款跳转----------------

    # --------------------- 判断两个列表中是否有相等----------------
    #  list1 查询list2
    def equal2list(self,list1,list2):
        flag = 0
        if len(list1)!=len(list2):
            for list in list1:
                if list in list2:
                    flag += 1
        else:
            if list1 == list2:
                flag = 1
        return flag


    def inequation(self,conditions, height, height_classification, sig):
        lst = []
        result = 0
        for condition in conditions:
            if '>' in condition or '<' in condition or '>=' in condition \
                    or "<=" in condition:
                equation = 'height' + condition
                result = eval(equation)
                if result == False:
                    index = 0
                else:
                    index = 1
                lst.append(index)
            else:
                index = self.equal2list(height_classification, condition)
                lst.append(index)
        if sig == '&':
            if 0 in lst:
                result = 0
            else:
                result = 1
        else:
            if any(lst) == 1:
                result = 1
            else:
                result = 0
        return result
    #高度分类属性判断  result = split_heightclasscation(rules['高度分类特性-建筑二'][0],40,["低层"])
    def split_heightclasscation(self,rule, height=None, height_classification=None):
        index = 0
        if '>' in rule or '<' in rule or '>=' in rule or "<=" in rule:
            if '&' in rule:
                conditions = rule.split('&')
                index = self.inequation(conditions, height, height_classification, '&')
            elif '|' in rule:
                conditions = rule.split('|')
                index = self.inequation(conditions, height, height_classification, '|')
            else:
                final = 'height' + rule
                r = eval(final)
                if r == False:
                    index = 0
                else:
                    index = 1
        else:
            # if '&' in rules['高度分类特性-建筑二']:
            #     result=rules['高度分类特性-建筑二'].split('&')
            #     if any(result)!=height_classification_2:
            #         print(0)
            if '|' in rule:
                classifications = rule.split('|')
                index = self.equal2list(height_classification,classifications)
            else:

                if rule not in height_classification:
                    index = 0
                else:
                    index = 1
        return index

    def jump_rule(self, rule):
        tmp_result = 0
        cal = 0
        final_result = 0
        for i in range(len(self.full_rules)):
            tmp_rule = self.full_rules[i]
            if tmp_rule['规范序号'] == rule:
                for j in range(len(tmp_rule['content'])):
                    self.rules = tmp_rule['content'][j]
                    tmp_result = self.mainrules_comp()
                    if tmp_result != 0:
                        final_result = tmp_result
                        cal += 1
                    else:
                        self.all_info.swap()
                        tmp_result = self.mainrules_comp()
                        if tmp_result != 0:
                            final_result = tmp_result
                            cal += 1
                if cal > 1:
                    raise ValueError("出现多个值，逻辑可能错误")
        return final_result

    def mainrules_comp(self):

        rules_lst=[l for l in self.rules.keys()]
        if '条款类型' in rules_lst:
            if self.rules['条款类型'] != ['检测条款']:
                return 0
        if '建筑属性-建筑一' in rules_lst:
            flag = 0
            for height in self.all_info.building_attr_1:
                if height in self.rules['建筑属性-建筑一']:
                    flag += 1
            if flag == 0:
                return 0

        if '建筑属性-建筑二' in rules_lst:
            if '建筑属性-建筑二' not in self.ban_rule:
                flag = 0
                for height in self.all_info.building_attr_2:
                    if height in self.rules['建筑属性-建筑二']:
                        flag += 1
                if flag == 0:
                    return 0
        if '高度分类特性-建筑一' in rules_lst:
            if '高度分类特性-建筑一' not in  self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['高度分类特性-建筑一'][0], self.all_info.building_high_1,self.all_info.height_classification_1):
                    return 0
        if '高度分类特性-建筑二' in rules_lst:
            if '高度分类特性-建筑二' not in self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['高度分类特性-建筑二'][0], self.all_info.building_high_2,self.all_info.height_classification_2):
                    return 0

        if '建筑高度-建筑一' in rules_lst:
            if '建筑高度-建筑一' not in self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['建筑高度-建筑一'][0], self.all_info.building_high_1):
                    return 0
        if '建筑高度-建筑二' in rules_lst:
            if '建筑高度-建筑二' not in self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['建筑高度-建筑二'][0], self.all_info.building_high_2):
                    return 0
        if '布置方式' in rules_lst:
            if '布置方式' not in self.ban_rule:
                if self.all_info.Arrangement_mode not in self.rules['布置方式']:
                    return 0
        if '建筑相对关系' in rules_lst:
            if '建筑相对关系' not in self.ban_rule:
                if self.all_info !=self.rules['建筑相对关系']:
                    return 0

        if '建筑夹角' in rules_lst:
            if '建筑夹角' not in self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['建筑夹角'][0], self.all_info.building_angle):
                    '''
                if '&' in self.rules['建筑夹角'][0]:
                    self.all_info.cal_building_angle()
                    left,right=self.rules['建筑夹角'][0].split('&')
                    left_value=str(self.all_info.building_angle)+left
                    right_value=str(self.all_info.building_angle)+right
                    if eval(left_value)==False or eval(right_value)==False:
                        return 0
                else:
                    self.all_info.cal_building_angle()
                    value=str(self.all_info.building_angle)+self.rules['建筑夹角'][0]
                    if eval(value)==False:
                    '''
                    return 0


        if '相对山墙设施' in rules_lst:
            if '相对山墙设施' not in self.ban_rule:
                relative_gable_facilities_list=[]
                relative_gable_facilities_list.append(self.all_info.relative_gable_facilities)
                if self.all_info.relative_gable_facilities =="双侧" or self.all_info.relative_gable_facilities =="单侧":
                    relative_gable_facilities_list.append('有')
                if 0 == self.equal2list(relative_gable_facilities_list,self.rules['相对山墙设施']):
                    return 0

        if '建筑朝向-建筑一' in rules_lst:
            if '建筑朝向-建筑一' not in self.ban_rule:
                flag =0
                for  orientation in self.all_info.main_orientation_1:
                    if orientation in self.rules['建筑朝向-建筑一']:
                        flag+=1
                if flag==0:
                    return 0
        if '建筑朝向-建筑二' in rules_lst:
            if '建筑朝向-建筑二' not in self.ban_rule:
                flag = 0
                for orientation in self.all_info.main_orientation_2:
                    if orientation in self.rules['建筑朝向-建筑二']:
                        flag+=1
                if flag==0:
                    return 0

        if '建筑位置-建筑一' in rules_lst:
            if '建筑位置-建筑一' not in self.ban_rule:
                if self.all_info.building_location_1 not in self.rules['建筑位置-建筑一']:
                    return 0

        if '建筑位置-建筑二' in rules_lst:
            if '建筑位置-建筑二' not in self.ban_rule:
                if self.all_info.building_location_2 not in  self.rules['建筑位置-建筑二']:
                    return 0
        if '山墙宽度-建筑一' in rules_lst:
            if '山墙宽度-建筑一' not in self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['山墙宽度-建筑一'][0], self.all_info.gable_width_1):
                    return 0
        if '山墙宽度-建筑二' in rules_lst:
            if '山墙宽度-建筑二' not in self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['山墙宽度-建筑二'][0], self.all_info.gable_width_2):
                    return 0

        if '相对山墙宽度' in rules_lst:
            if '相对山墙宽度' not in self.ban_rule:
                #self.all_info.cal_gabel_width()
                w = self.all_info.relative_gable_width
                c = 'w'+self.rules['相对山墙宽度'][0]
                r = eval(c)
                if r == False:
                    return 0

        if '山墙设施-建筑一' in rules_lst:
            if '山墙设施-建筑一' not in self.ban_rule:
                if self.all_info.gable_facilities_1 not in self.rules['山墙设施-建筑一']:
                    return 0

        if '山墙设施-建筑二' in rules_lst:
            if '山墙设施-建筑二' not in self.ban_rule:
                if self.all_info.gable_facilities_2 not in self.rules['山墙设施-建筑二']:
                    return 0

        if '建筑形态-建筑一' in rules_lst:
            if '建筑形态-建筑一' not in self.ban_rule:
                if self.all_info.building_form_1 not in self.rules['建筑形态-建筑一']:
                    return 0

        if '建筑形态-建筑二' in rules_lst:
            if '建筑形态-建筑二' not in self.ban_rule:
                if self.all_info.building_form_2 not in self.rules['建筑形态-建筑二']:
                    return 0

        if '建筑面宽-建筑一' in rules_lst:
            if '建筑面宽-建筑一' not in self.ban_rule:
                if self.all_info.building_width_1 !=self.rules['建筑面宽-建筑一']:
                    return 0

        if '建筑面宽-建筑二' in rules_lst:
            if '建筑面宽-建筑二' not in self.ban_rule:
                if self.all_info.building_width_2 !=self.rules['建筑面宽-建筑二']:
                    return 0

        if '山墙投影宽度' in rules_lst:
            if '山墙投影宽度' not in self.ban_rule:
                if self.all_info.gable_projection_width !=self.rules['山墙投影宽度']:
                    return 0

        if '正向重叠长度' in rules_lst:
            if '正向重叠长度' not in self.ban_rule:
                if 0 == self.split_heightclasscation(self.rules['正向重叠长度'][0], self.all_info.forward_overlap_length):
                    return 0

        if '首层性质-建筑一' in rules_lst:
            if '首层性质-建筑一' not in self.ban_rule:
                if self.building1 !=self.rules['首层性质-建筑一']:
                    return 0

        if '首层性质-建筑二' in rules_lst:
            if '首层性质-建筑二' not in self.ban_rule:
                if self.building2 !=self.rules['首层性质-建筑二']:
                    return 0

        if '首层高度-建筑一' in rules_lst:
            if '首层高度-建筑一' not in self.ban_rule:
                if self.building1 !=self.rules['首层高度-建筑一']:
                    return 0

        if '首层高度-建筑二' in rules_lst:
            if '首层高度-建筑二' not in self.ban_rule:
                if self.building2 !=self.rules['首层高度-建筑二']:
                    return 0

        if '首层架空-建筑一' in rules_lst:
            if '首层架空-建筑一' not in self.ban_rule:
                if self.building1 !=self.rules['首层架空-建筑一']:
                    return 0

        if '首层架空-建筑二' in rules_lst:
            if '首层架空-建筑二' not in self.ban_rule:
                if self.building2 !=self.rules['首层架空-建筑二']:
                    return 0

        if '底层非住宅高度-建筑一' in rules_lst:
            if '底层非住宅高度-建筑一' not in self.ban_rule:
                if self.building1 !=self.rules['底层非住宅高度-建筑一']:
                    return 0

        if '底层非住宅高度-建筑二' in rules_lst:
            if '底层非住宅高度-建筑二' not in self.ban_rule:
                if self.building2 !=self.rules['底层非住宅高度-建筑二']:
                    return 0
        if '区域类型' in rules_lst:
            if '区域类型' not in self.ban_rule:
                if self.all_info.area_type not in self.rules['区域类型']:
                    return 0


        if '公式' in rules_lst or '最小值' in rules_lst :
            result = 0
            if '公式' not in self.ban_rule and '公式' in rules_lst:
                if self.rules['公式']['公式']!='':
                    if 'Hs' in self.rules['公式']['公式']:
                        if self.all_info.y_cord_2 >= self.all_info.y_cord_1:
                            Hs = self.all_info.building_high_2
                        else:
                            Hs = self.all_info.building_high_1
                        result=eval(self.rules['公式']['公式'])
                    elif 'Hp' in self.rules['公式']['公式']:
                        Hp = (self.all_info.building_high_2 + self.all_info.building_high_1) / 2
                        result=eval(self.rules['公式']['公式'])
                    elif 'Hg' in self.rules['公式']['公式']:
                        Hg=max(self.all_info.building_high_2,self.all_info.building_high_1)
                        result=eval(self.rules['公式']['公式'])
                    elif 'H' not in self.rules['公式']['公式'] :
                        result = float(self.rules['公式']['公式'])
                    else:
                        H=max(self.all_info.building_high_2,self.all_info.building_high_1)
                        result=eval(self.rules['公式']['公式'])

                elif self.rules['公式']['跳转']!=[]:
                    if self.rules['公式']['跳转系数']!='':
                        index=float(self.rules['公式']['跳转系数'])
                    bounding=0
                    if '最小值' in rules_lst:
                        bounding=float(self.rules['最小值'][0])
                    if "跳转忽略筛选项" in rules_lst:
                        self.ban_rule=self.rules['跳转忽略筛选项']
                        for a in self.rules['公式']['跳转']:
                            result=index*self.jump_rule(a)
                            if result !=0:
                                if '最小值' in rules_lst:
                                    result = max(result, bounding)
                                break
                        self.ban_rule = ''

                elif self.rules['公式']['消防间距'] !='':
                    result = max(result, float(self.rules['公式']['消防间距']))
                elif self.rules['公式']['其他'] !='':
                    result = max(result, float(self.rules['公式']['其他']))
            if '最小值' in rules_lst:
                result = max(result, float(self.rules['最小值'][0]))

            return result
        else:
            return 0


    def building_rules_trave(self):
        lst=[]
        count = 0
        rule_num = len(self.full_rules)
        for i in range(rule_num):
            for j in range(len(self.full_rules[i]['content'])):
                self.rules = self.full_rules[i]['content'][j]
                dist = self.mainrules_comp()

                if dist != 0:
                    print("匹配条款",i,j)
                    lst.append(dist)
                    count += 1
                else:
                    self.all_info.swap()
                    dist = self.mainrules_comp()
                    if dist != 0:
                        print("匹配条款", i, j)
                        lst.append(dist)
                        count += 1
        # if count>1:
        #     raise ValueError('条款逻辑错误，出现多个值,')
        # elif count==0:
        #     raise ValueError('条款逻辑错误,都不符合')
        return count, lst




