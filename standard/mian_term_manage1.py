

import math
import sys

#from building_info import input_A_B
#from building_pre import input_pre

class main_term_mannage:

    def __init__(self, json_path_pre,building_1,building_2):

        # --------------------- 输入变量 ---------------------
        #self.json_main_excel = json_path_excel
        self.json_pre = json_path_pre
        self.building_one=building_1
        self.building_two = building_2
        # --------------------- 主条款属性 ---------------------
        # --------------------- from pre ---------------------
        # 建筑属性一/二:["居住建筑","非居住建筑"]
        self.building_attr_1 = []
        self.building_attr_2 = []
        # 高度分类特征一/二;["低层","多层","中高层","高层"]
        self.height_classification_1 = []
        self.height_classification_2 = []
        # 布置方式:"平行"
        self.Arrangement_mode = " "
        # 建筑朝向一/二:"南北向"
        self.main_orientation_1 = []
        self.main_orientation_2 = []
        # 区域类型:"新区","旧区"
        self.area_type = " "
        # 建筑位置一/二:["南","北","东","西"]
        self.building_location_1 = " "
        self.building_location_2 = " "
        # 正向重叠长度:16.0
        self.forward_overlap_length = 0.0
        # 建筑形态一/二:["点式","条式"]
        self.building_form_1 = " "
        self.building_form_2 = " "
        # # --------------------- from json_main_excel ---------------------
        # # 规范序号:5.2
        # self.norm_number = 0.0
        # # 公式
        # self.formula = {}
        # # 最小值:0.0
        # self.min_value = 0.0
        # --------------------- from A、B ---------------------
        # 建筑高度:35.0
        self.building_high_1 = 0.0
        self.building_high_2 = 0.0
        # 建筑一/二中心坐标:35.0
        self.x_cord_1 = 0.0
        self.y_cord_1 = 0.0
        self.x_cord_2 = 0.0
        self.y_cord_2 = 0.0
        #建筑A、B角度
        self.main_direction_1 = 0.0
        self.main_direction_2 = 0.0
        # 建筑夹角:30.0
        self.building_angle = 0.0
        # 相对山墙设施:"有"
        self.relative_gable_facilities = " "
        # 山墙设施一/二:["有","无","单侧","两侧"]
        self.gable_facilities_1 = " "
        self.gable_left_1=0
        self.gable_right_1 = 0
        self.gable_facilities_2 = " "
        self.gable_left_2=0
        self.gable_right_2 = 0
        # 山墙宽度一/二:16.0
        self.gable_width_1 = 0.0
        self.gable_width_2 = 0.0
        # 相对山墙宽度:16.0
        self.relative_gable_width = 0.0

        # 建筑面宽一/二(长边宽):32.0
        self.building_width_1 = 0.0
        self.building_width_2 = 0.0
        # 山墙投影宽度:12.0
        self.gable_projection_width = 0.0
        # 建筑相对关系:["主朝向相对","山墙相对","主朝向对山墙","不相对"]
        self.building_relation = " "
        # 首层性质一/二
        self.first_floor_property_1 = ""
        self.first_floor_property_2 = ""
        # 首层高度一/二
        self.first_floor_height_1 = 0.0
        self.first_floor_height_2 = 0.0
        # 首层架空一/二
        self.first_floor_overhead_1 = 0.0
        self.first_floor_overhead_2 = 0.0
        # 底层非住宅高度一/二
        self.ground_floor_non_resdential_height_1 = 0.0
        self.ground_floor_non_resdential_height_2 = 0.0
        # 输出间距
        self.building_gap = []
        # --------------------- 方法 ---------------------

        self.read_A_data()
        self.read_B_data()
        self.read_pre_data()

        self.judge_gabel_facilities()
        self.judge_relation_gabel_facilities()
        self.cal_gabel_width()
        self.cal_building_angle()





    def read_A_data(self):
        self.building_high_1 = self.cal_height(self.building_one)
        self.gable_width_1 = self.building_one["depth"]
        self.building_width_1 = self.building_one["width"]
        self.x_cord_1 = self.building_one["coordinate"][0]
        self.y_cord_1 = self.building_one["coordinate"][1]
        self.main_direction_1=float(self.building_one['main_direction'])
        self.gable_right_1 = self.building_one["gable_right"]
        self.gable_left_1 = self.building_one["gable_left"]


    def read_B_data(self):
        self.building_high_2 = self.cal_height(self.building_two)
        self.gable_width_2 = self.building_two["depth"]
        self.building_width_2 = self.building_two["width"]
        self.x_cord_2 = self.building_two["coordinate"][0]
        self.y_cord_2 = self.building_two["coordinate"][1]
        self.main_direction_2 = float(self.building_two['main_direction'])
        self.gable_right_2 = self.building_two["gable_right"]
        self.gable_left_2 = self.building_two["gable_left"]

    def read_pre_data(self):
        self.building_attr_1 = self.json_pre["建筑属性-建筑一"]
        self.building_attr_2 = self.json_pre["建筑属性-建筑二"]
        self.height_classification_1 = (self.json_pre["高度分类特性-建筑一"])
        self.height_classification_2 = (self.json_pre["高度分类特性-建筑二"])
        self.main_orientation_1 = (self.json_pre["建筑朝向-建筑一"])
        self.main_orientation_2 = (self.json_pre["建筑朝向-建筑二"])
        self.building_location_1 = (self.json_pre['建筑位置-建筑一'])
        self.building_location_2 = self.json_pre['建筑位置-建筑二']

        self.area_type = self.json_pre["区域类型"]
        self.Arrangement_mode = self.json_pre["布置方式"]
        self.forward_overlap_length = self.json_pre['正向重叠长度']
        self.building_form_1 = self.json_pre['建筑形态-建筑一']
        self.building_form_2 = self.json_pre['建筑形态-建筑二']

    def opppose_read_pre_data(self):
        self.building_attr_2,self.building_attr_1 = self.building_attr_1,self.building_attr_2
        self.height_classification_2,self.height_classification_1 = self.height_classification_1 ,self.height_classification_2
        self.main_orientation_2,self.main_orientation_1 = self.main_orientation_1,self.main_orientation_2
        self.building_location_2 ,self.building_location_1 = self.building_location_1,self.building_location_2
        self.building_form_2, self.building_form_1 = self.building_form_1, self.building_form_2
        # self.area_type = self.area_type
        # self.Arrangement_mode = self.Arrangement_mode

    # --------------------- 判断有无山墙设施 ---------------------
    def judge_gabel_facilities(self):
        if self.gable_right_1 == 1 or self.gable_left_1 == 1:
            self.gable_facilities_1 = "有"
        else:
            self.gable_facilities_1 = "无"
        if self.gable_right_2 == 1 or self.gable_left_2 == 1:
            self.gable_facilities_2 = "有"
        else:
            self.gable_facilities_2 = "无"

    # --------------------- 判断相对墙设施 ---------------------
    def judge_relation_gabel_facilities(self):
        if self.Arrangement_mode == "平行":
            if self.main_orientation_1 == "南北向":
                if self.building_location_1 == "南":
                    self.relative_gable_facilities = "无"
                elif self.building_location_1 == "北":
                    self.relative_gable_facilities = "无"
                    #山墙相对
                elif self.building_location_1 == "东" :
                    self.judge_gabel_sate(self.gable_left_1, self.gable_right_2)
                elif self.building_location_1 == "西":
                    self.judge_gabel_sate(self.gable_left_2, self.gable_right_1)
            if self.main_orientation_1 == "东西向":
                if self.building_location_1 == "东":
                    self.relative_gable_facilities = "无"
                elif self.building_location_1 == "西":
                    self.relative_gable_facilities = "无"
                    #山墙相对
                elif self.building_location_1 == "南":
                    self.judge_gabel_sate(self.gable_left_2, self.gable_right_1)
                elif self.building_location_1 == "北":
                    self.judge_gabel_sate(self.gable_left_1, self.gable_right_2)
        elif self.Arrangement_mode == "垂直" or self.Arrangement_mode == "非平行非垂直":
            if self.main_direction_1 < self.main_direction_2:
                if self.building_location_1 == "南" :
                    self.judge_gabel_sate(self.gable_left_2)
                elif self.building_location_1 == "北" :
                    self.judge_gabel_sate(self.gable_right_2)
                elif self.building_location_1 == "东" :
                    self.judge_gabel_sate(self.gable_left_1)
                elif self.building_location_1 == "西":
                    self.judge_gabel_sate(self.gable_right_1)
            else:
                if self.building_location_1 == "南" :
                    self.judge_gabel_sate(self.gable_left_1)
                elif self.building_location_1 == "北" :
                    self.judge_gabel_sate(self.gable_right_1)
                elif self.building_location_1 == "东" :
                    self.judge_gabel_sate(self.gable_right_2)
                elif self.building_location_1 == "西":
                    self.judge_gabel_sate(self.gable_left_2)
        elif self.Arrangement_mode == "山墙相对":
            if "南北向" in self.main_orientation_1:
                if self.x_cord_1 > self.x_cord_2:
                    self.judge_gabel_sate(self.gable_left_1, self.gable_right_2)
                else:
                    self.judge_gabel_sate(self.gable_left_2, self.gable_right_1)
            if "东西向" in self.main_orientation_1:
                if self.y_cord_1 > self.y_cord_2:
                    self.judge_gabel_sate(self.gable_left_2, self.gable_right_1)
                else:
                    self.judge_gabel_sate(self.gable_left_1, self.gable_right_2)




    # --------------------- 判断相对山墙状态 ---------------------
    def judge_gabel_sate(self, gable_1, gable_2=None):

        if gable_1 == 1 and gable_2 == 0:
            self.relative_gable_facilities = "单侧"
        elif gable_1 == 0 and gable_2 == 1:
            self.relative_gable_facilities = "单侧"
        elif gable_1 == 1 and gable_2 == 1:
            self.relative_gable_facilities = "双侧"
        else:
            self.relative_gable_facilities = "无"

    # --------------------- 计算相对山墙宽度 ---------------------
    def cal_gabel_width(self):
        if self.Arrangement_mode == "垂直":
            if abs(self.main_direction_2) < abs(self.main_direction_1):
                if self.building_location_2 in ['南','北']:
                    self.relative_gable_width = self.gable_width_1
                if self.building_location_2 in ['东','西']:
                    self.relative_gable_width = self.gable_width_2
            else:
                if self.building_location_2 in ['南','北']:
                    self.relative_gable_width = self.gable_width_2
                if self.building_location_2 in ['东','西']:
                    self.relative_gable_width = self.gable_width_1

    # --------------------- 计算楼房高度 ---------------------
    def cal_height(self,building):

        return float(building['storey_height']*building['layers'])
    # --------------------- 计算建筑夹角 ---------------------
    def cal_building_angle(self):
        self.building_angle=abs(self.main_direction_1-self.main_direction_2)
    # --------------------- 计算山墙投影宽度 ---------------------
    def cal_gable_projection_width(self):
        return 0
    # --------------------- 计算正向重叠长度 ---------------------
    def cal_forward_overlap_length(self):
        return 0
    #------------------交换A、B所有信息----------------------------
    def swap(self):
        self.building_one ,self.building_two = self.building_two ,self.building_one
        self.read_A_data()
        self.read_B_data()
        self.opppose_read_pre_data()
        self.judge_gabel_facilities()
        self.judge_relation_gabel_facilities()
        self.cal_gabel_width()
        self.cal_building_angle()

