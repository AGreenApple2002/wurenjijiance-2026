#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77 
@File    ：start_window.py
@IDE     ：PyCharm
@Description  ：主要的图形化界面，本次图形化界面实现的主要技术为pyside6，pyside6是官方提供支持的
'''
import json
import copy                      # 用于图像复制
import os                        # 用于系统路径查找
import shutil                    # 用于复制
# from distutils.command.config import config
from PySide6.QtGui import *      # GUI组件
from PySide6.QtCore import *     # 字体、边距等系统变量
from PySide6.QtWidgets import *  # 窗口等小组件

from PySide6.QtWidgets import QWidget, QFormLayout, QLineEdit, QVBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt

import threading                 # 多线程
import sys                       # 系统库
import cv2                       # opencv图像处理
import torch                     # 深度学习框架
import os.path as osp            # 路径查找
import time                      # 时间计算
from ultralytics import YOLO     # yolo核心算法
from ultralytics.utils.torch_utils import select_device
from collections import defaultdict, UserDict
import numpy as np
# 常用的字符串常量
WINDOW_TITLE ="Target detection system"            # 系统上方标题
WELCOME_SENTENCE = "基于无人机实时航拍的目标智能检测与识别系统"   # 欢迎的句子
ICON_IMAGE = "images/UI/777.png"                 # 系统logo界面
IMAGE_LEFT_INIT = "images/UI/img_1.png"              # 图片检测界面初始化左侧图像
IMAGE_RIGHT_INIT = "images/UI/img_1.png"          # 图片检测界面初始化右侧图像
ZHU_IMAGE_PATH = "images/UI/zhu.jpg"
USERNAME = "123"
PASSWORD = "123"
LOGIN_TITLE = "智巡空域，精准识别"

class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)       # 系统界面标题
        self.resize(1200, 800)                  # 系统初始化大小
        self.setWindowIcon(QIcon(ICON_IMAGE))   # 系统logo图像
        self.output_size = 480                  # 上传的图像和视频在系统界面上显示的大小
        self.img2predict = ""                   # 要进行预测的图像路径
        # 用来进行设置的参数
        self.init_vid_id = '0'                  # 网络摄像头修改 包括ip或者是ip地址的修改
        self.vid_source = int(self.init_vid_id) # 需要设置为对应的整数，加载的才是usb的摄像头
        self.conf_thres = 0.25   # 置信度的阈值
        self.iou_thres = 0.45    # NMS操作的时候 IOU过滤的阈值
        self.save_txt = False
        self.save_conf = False
        self.save_crop = False
        self.vid_gap = 30        # 摄像头视频帧保存间隔。
        self.is_open_track = ""  # 三种选择，如果是空表示不开启追踪，否则有两种追踪器可以进行选择


        self.cap = cv2.VideoCapture(self.vid_source)
        self.video_writer = None
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        # self.model_path = "runs/detect/yolo11-n/weights/best.pt"  # todo 指明模型加载的位置的设备
        # self.model_path = "runs/detect/DroneVehicle/train7/weights/best.pt"  # todo 指明模型加载的位置的设备
        #self.model_path = "runs/detect/MyVisDrone/myvisdrone_exp12/weights/best.pt"
        self.model_path = "runs/detect/MyVisDrone_train100/weights/best.pt"
        self.model = self.model_load(weights=self.model_path)
        #加入的新代码 共三行
        # self.config_track_value = QComboBox(self)
        # self.config_track_value.addItems(['不开启追踪', "bytetrack.yaml", "botsort.yaml"])
        # self.config_track_value.setCurrentText('不开启追踪')
        # 初始化其他可能被引用的配置控件
        self.config_output_size_value = QLineEdit(str(self.output_size))
        self.config_vid_source_value = QLineEdit(str(self.vid_source))
        self.config_vid_gap_value = QLineEdit(str(self.vid_gap))
        self.config_conf_thres_value = QLineEdit(str(self.conf_thres))
        self.config_iou_thres_value = QLineEdit(str(self.iou_thres))
     #到这里结束
        self.initUI()            # 初始化图形化界面
        self.reset_vid()         # 重新设置视频参数，重新初始化是为了防止视频加载出错

    # 模型初始化
    @torch.no_grad()
    def model_load(self, weights=""):
        """
        模型加载
        """
        # 模型加载的时候配合置信度一起使用
        model_loaded = YOLO(weights)
        return model_loaded

    def apply_modern_theme(self):
        """Apply a cleaner visual theme."""
        self.setObjectName("mainTabs")
        self.setStyleSheet(
            """
            QTabWidget#mainTabs::pane {
                border: 1px solid #d7dee7;
                border-radius: 18px;
                background: #f7f9fc;
                top: -1px;
            }
            QTabBar::tab {
                background: #e7edf5;
                color: #475569;
                border: 1px solid #d7dee7;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                padding: 10px 22px;
                margin-right: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0f172a;
                border-bottom-color: #ffffff;
            }
            QTabBar::tab:hover {
                background: #dce8f3;
            }
            QWidget {
                background: #eef3f8;
                color: #1f2937;
            }
            QLabel {
                background: transparent;
            }
            QWidget#panelCard {
                background: #ffffff;
                border: 1px solid #dde5ee;
                border-radius: 18px;
            }
            QLabel#sectionTitle {
                color: #0f172a;
                font-size: 18px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#resultLabel, QLabel#infoLabel {
                background: #f8fbff;
                color: #334155;
                border: 1px solid #dbe5f0;
                border-radius: 12px;
                padding: 12px 14px;
            }
            QLabel#previewFrame {
                background: #f8fafc;
                border: 1px solid #dbe3ec;
                border-radius: 16px;
                padding: 10px;
            }
            QPushButton {
                border: none;
                border-radius: 12px;
                padding: 12px 18px;
                font-size: 14px;
                font-weight: 600;
                min-height: 18px;
            }
            QPushButton#primaryButton {
                background: #0f6cbd;
                color: #ffffff;
            }
            QPushButton#primaryButton:hover {
                background: #115ea3;
            }
            QPushButton#primaryButton:pressed {
                background: #0c4f8a;
            }
            QPushButton#secondaryButton {
                background: #e8f0f8;
                color: #17406d;
                border: 1px solid #c8d9eb;
            }
            QPushButton#secondaryButton:hover {
                background: #dbe9f6;
            }
            QPushButton#secondaryButton:pressed {
                background: #cfdff0;
            }
            QPushButton#dangerButton {
                background: #d94f4f;
                color: #ffffff;
            }
            QPushButton#dangerButton:hover {
                background: #bb3f3f;
            }
            QPushButton#dangerButton:pressed {
                background: #9f3333;
            }
            QPushButton:disabled {
                background: #cfd8e3;
                color: #7b8794;
            }
            QLineEdit, QComboBox {
                background: #ffffff;
                border: 1px solid #d6dee8;
                border-radius: 10px;
                padding: 8px 10px;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #0f6cbd;
            }
            QRadioButton {
                spacing: 8px;
                color: #334155;
            }
            """
        )

    def set_scaled_pixmap(self, label, image_path):
        """Scale preview images to the visible label area without cropping."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        target_size = label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = QSize(self.output_size, self.output_size)
        scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def initUI(self):
        """
        图形化界面初始化
        """
        # ********************* 图片识别界面 *****************************
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        img_detection_widget = QWidget()
        img_detection_widget.setObjectName("panelCard")
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片检测功能")
        img_detection_title.setObjectName("sectionTitle")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_widget.setObjectName("panelCard")
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setObjectName("previewFrame")
        self.right_img.setObjectName("previewFrame")
        # self.left_img.setFixedSize(520, 520)
        # self.right_img.setFixedSize(520, 520)
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addWidget(self.right_img)
        self.img_num_label = QLabel("当前检测结果：待检测")
        self.img_num_label.setObjectName("resultLabel")
        self.img_num_label.setFont(font_main)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.setObjectName("secondaryButton")
        det_img_button.setObjectName("primaryButton")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        # up_img_button.setStyleSheet("QPushButton{color:white}"
        #                             "QPushButton:hover{background-color: rgb(2,110,180);}"
        #                             "QPushButton{background-color:rgb(48,124,208)}"
        #                             "QPushButton{border:2px}"
        #                             "QPushButton{border-radius:5px}"
        #                             "QPushButton{padding:5px 5px}"
        #                             "QPushButton{margin:5px 5px}")
        # det_img_button.setStyleSheet("QPushButton{color:white}"
        #                              "QPushButton:hover{background-color: rgb(2,110,180);}"
        #                              "QPushButton{background-color:rgb(48,124,208)}"
        #                              "QPushButton{border:2px}"
        #                              "QPushButton{border-radius:5px}"
        #                              "QPushButton{padding:5px 5px}"
        #                              "QPushButton{margin:5px 5px}")
        img_detection_layout.setContentsMargins(20, 20, 20, 20)
        img_detection_layout.setSpacing(16)
        mid_img_layout.setContentsMargins(16, 16, 16, 16)
        mid_img_layout.setSpacing(16)
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.img_num_label)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # ********************* 视频识别界面 *****************************
        vid_detection_widget = QWidget()
        vid_detection_widget.setObjectName("panelCard")
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测功能")
        vid_title.setObjectName("sectionTitle")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setObjectName("previewFrame")
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setObjectName("secondaryButton")
        self.mp4_detection_btn.setObjectName("primaryButton")
        self.vid_stop_btn.setObjectName("dangerButton")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        # self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
        #                                         "QPushButton:hover{background-color: rgb(2,110,180);}"
        #                                         "QPushButton{background-color:rgb(48,124,208)}"
        #                                         "QPushButton{border:2px}"
        #                                         "QPushButton{border-radius:5px}"
        #                                         "QPushButton{padding:5px 5px}"
        #                                         "QPushButton{margin:5px 5px}")
        # self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
        #                                      "QPushButton:hover{background-color: rgb(2,110,180);}"
        #                                      "QPushButton{background-color:rgb(48,124,208)}"
        #                                      "QPushButton{border:2px}"
        #                                      "QPushButton{border-radius:5px}"
        #                                      "QPushButton{padding:5px 5px}"
        #                                      "QPushButton{margin:5px 5px}")
        # self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
        #                                 "QPushButton:hover{background-color: rgb(2,110,180);}"
        #                                 "QPushButton{background-color:rgb(48,124,208)}"
        #                                 "QPushButton{border:2px}"
        #                                 "QPushButton{border-radius:5px}"
        #                                 "QPushButton{padding:5px 5px}"
        #                                 "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        vid_detection_layout.setContentsMargins(20, 20, 20, 20)
        vid_detection_layout.setSpacing(16)
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        # todo 添加摄像头检测标签逻辑
        self.vid_num_label = QLabel("当前检测结果：{}".format("等待检测"))
        self.vid_num_label.setObjectName("resultLabel")
        self.vid_num_label.setFont(font_main)
        vid_detection_layout.addWidget(self.vid_num_label)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # ********************* 模型切换界面 *****************************
        about_widget = QWidget()
        about_widget.setObjectName("panelCard")
        about_layout = QVBoxLayout()
        about_title = QLabel(WELCOME_SENTENCE)
        about_title.setFont(QFont('楷体', 18))
        about_title.setObjectName("sectionTitle")
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setObjectName("previewFrame")
        about_img.setPixmap(QPixmap(ZHU_IMAGE_PATH))
        self.model_label = QLabel("当前模型：{}".format(self.model_path))
        self.model_label.setObjectName("infoLabel")
        self.model_label.setFont(font_main)
        change_model_button = QPushButton("切换模型")
        change_model_button.setObjectName("primaryButton")
        change_model_button.setFont(font_main)

        record_button = QPushButton("查看历史记录")
        record_button.setObjectName("secondaryButton")
        record_button.setFont(font_main)
        record_button.clicked.connect(self.check_record)
        change_model_button.clicked.connect(self.change_model)
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.setContentsMargins(20, 20, 20, 20)
        about_layout.setSpacing(16)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addWidget(self.model_label)
        about_layout.addStretch()
        about_layout.addWidget(change_model_button)
        about_layout.addWidget(record_button)
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        self.left_img.setAlignment(Qt.AlignCenter)

        # ********************* 配置切换界面 ****************************

        config_widget = QWidget()
        config_widget.setObjectName("panelCard")

        config_grid_widget = QWidget()
        config_grid_widget.setObjectName("panelCard")
        config_grid_layout = QFormLayout()

        # self.output_size = 480  # 上传的图像和视频在系统界面上显示的大小
        config_output_size_label = QLabel("系统图像显示大小")
        self.config_output_size_value = QLineEdit("")
        self.config_output_size_value.setText(str(self.output_size))
        config_grid_layout.addRow(config_output_size_label, self.config_output_size_value)


        # # 用来进行设置的参数
        # self.init_vid_id = '0'  # 网络摄像头修改 包括ip或者是ip地址的修改
        config_vid_source_label = QLabel("摄像头源地址")
        self.config_vid_source_value = QLineEdit("")
        self.config_vid_source_value.setText(str(self.vid_source))
        config_grid_layout.addRow(config_vid_source_label, self.config_vid_source_value)

        # self.vid_gap = 30  # 摄像头视频帧保存间隔。
        config_vid_gap_label = QLabel("视频帧保存间隔")
        self.config_vid_gap_value = QLineEdit("")
        self.config_vid_gap_value.setText(str(self.vid_gap))
        config_grid_layout.addRow(config_vid_gap_label, self.config_vid_gap_value)

        # self.vid_source = int(self.init_vid_id)  # 需要设置为对应的整数，加载的才是usb的摄像头
        # self.conf_thres = 0.25  # 置信度的阈值
        config_conf_thres_label = QLabel("检测模型置信度阈值")
        self.config_conf_thres_value = QLineEdit("")
        self.config_conf_thres_value.setText(str(self.conf_thres))
        config_grid_layout.addRow(config_conf_thres_label, self.config_conf_thres_value)

        # self.iou_thres = 0.45  # NMS操作的时候 IOU过滤的阈值
        config_iou_thres_label = QLabel("检测模型IOU阈值")
        self.config_iou_thres_value = QLineEdit("")
        self.config_iou_thres_value.setText(str(self.iou_thres))
        config_grid_layout.addRow(config_iou_thres_label, self.config_iou_thres_value)

        # 追踪配置
        config_track_label = QLabel("追踪配置")
        self.config_track_value = QComboBox(self)
        # results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        # results = model.track(frame, persist=True, tracker="botsort.yaml")
        self.config_track_value.addItems(['不开启追踪', "bytetrack.yaml", "botsort.yaml"])
        config_grid_layout.addRow(config_track_label, self.config_track_value)
        # self.cb = QComboBox(self)
        # self.cb.move(100, 20)
        #
        # # 单个添加条目
        # self.cb.addItem('C')
        # self.cb.addItem('C++')
        # self.cb.addItem('Python')
        # # 多个添加条目
        # self.cb.addItems(['Java', 'C#', 'PHP'])

        # 追踪模型选择，以及是否使用追踪模型

        config_grid_layout.setContentsMargins(18, 18, 18, 18)
        config_grid_layout.setHorizontalSpacing(16)
        config_grid_layout.setVerticalSpacing(14)
        config_grid_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        config_grid_layout.setFormAlignment(Qt.AlignTop)
        config_grid_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        config_grid_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        config_grid_widget.setLayout(config_grid_layout)
        config_grid_widget.setFont(font_main)
        config_grid_widget.setMinimumWidth(900)
        
        for widget in (
            self.config_output_size_value,
            self.config_vid_source_value,
            self.config_vid_gap_value,
            self.config_conf_thres_value,
            self.config_iou_thres_value,
            self.config_track_value,
        ):
            widget.setMinimumWidth(540)
            widget.setMaximumWidth(960)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        save_config_button = QPushButton("保存配置信息")
        save_config_button.setObjectName("primaryButton")
        save_config_button.setFont(font_main)
        save_config_button.clicked.connect(self.save_config_change)
        # save_config_button.setStyleSheet("QPushButton{color:white}"
        #                             "QPushButton:hover{background-color: rgb(2,110,180);}"
        #                             "QPushButton{background-color:rgb(48,124,208)}"
        #                             "QPushButton{border:2px}"
        #                             "QPushButton{border-radius:5px}"
        #                             "QPushButton{padding:5px 5px}"
        #                             "QPushButton{margin:5px 5px}")
        config_layout = QVBoxLayout()
        config_vid_title = QLabel("配置信息修改功能")
        config_icon_label = QLabel()
        config_icon_label.setObjectName("previewFrame")
        config_icon_label.setMinimumWidth(620)
        config_icon_label.setFixedHeight(220)
        config_icon_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        config_icon_label.setPixmap(
            QPixmap("images/UI/config.png").scaled(
                QSize(620, 220), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        config_icon_label.setAlignment(Qt.AlignCenter)
        config_vid_title.setObjectName("sectionTitle")
        config_vid_title.setAlignment(Qt.AlignCenter)
        config_vid_title.setFont(font_title)
        config_layout.setContentsMargins(20, 20, 20, 20)
        config_layout.setSpacing(16)
        config_layout.addWidget(config_vid_title)
        config_layout.addWidget(config_icon_label)
        config_layout.addWidget(config_grid_widget)
        config_layout.addStretch()
        config_layout.addWidget(save_config_button)
        config_widget.setLayout(config_layout)


        # self.addTab(about_widget, '主页')
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        self.addTab(config_widget, '配置信息')
        self.setTabIcon(0, QIcon(ICON_IMAGE))
        self.setTabIcon(1, QIcon(ICON_IMAGE))
        self.setTabIcon(2, QIcon(ICON_IMAGE))
        self.setTabIcon(3, QIcon(ICON_IMAGE))
        self.apply_modern_theme()
        self.set_scaled_pixmap(self.left_img, IMAGE_LEFT_INIT)
        self.set_scaled_pixmap(self.right_img, IMAGE_RIGHT_INIT)
        self.set_scaled_pixmap(self.vid_img, "images/UI/up.jpeg")

        # ********************* todo 布局修改和颜色变换等相关插件 *****************************

    def upload_img(self):
        """上传图像，图像要尽可能保证是中文格式"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg') # 选择图像
        if fileName: # 如果存在文件名称则对图像进行处理
            # 将图像转移到当前目录下，解决中文的问题
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)  # 将图像转移到images目录下并且修改为英文的形式
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            # 调整图像的尺寸，让图像可以适应图形化的界面
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = save_path                               # 给变量进行赋值方便后面实际进行读取
            # 将图像显示在界面上并将预测的文字内容进行初始化
            self.set_scaled_pixmap(self.left_img, "images/tmp/upload_show_result.jpg")
            self.set_scaled_pixmap(self.right_img, IMAGE_RIGHT_INIT)
            self.img_num_label.setText("当前检测结果：待检测")

    def change_model(self):
        """切换模型，重新对self.model进行赋值"""
        # 用于pt格式模型的结果，这个模型必须是经过这里的代码训练出来的
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        if fileName:
            # 如果用户选择了对应的pt文件，根据用户选择的pt文件重新对模型进行初始化
            self.model_path = fileName
            self.model = self.model_load(weights=self.model_path)
            QMessageBox.information(self, "成功", "模型切换成功！")
            self.model_label.setText("当前模型：{}".format(self.model_path))

    # 图片检测
    def detect_img(self):
        """检测单张的图像文件"""
        output_size = self.output_size
        # model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)
        # self.save_txt = False
        #         self.save_conf = False
        #         self.save_crop = False
        print(self.save_txt)
        results = self.model(self.img2predict, conf=self.conf_thres, iou=self.iou_thres, save_txt=self.save_txt, save_conf=self.save_conf, save_crop=self.save_crop)  # 读取图像并执行检测的逻辑
        # 如果你想要对结果进行单独的解析请使用下面的内容
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        # 显示并保存检测的结果
        result = results[0]                     # 获取检测结果
        img_array = result.plot()               # 在图像上绘制检测结果
        im0 = img_array
        im_record = copy.deepcopy(im0)
        resize_scale = output_size / im0.shape[0]
        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        self.set_scaled_pixmap(self.right_img, "images/tmp/single_result.jpg")
        time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
        cv2.imwrite("record/img/{}.jpg".format(time_re), im_record)
        # 显示每个类别中检测出来的样本数量
        result_names = result.names
        result_nums = [0 for i in range(0, len(result_names))]
        cls_ids = list(result.boxes.cls.cpu().numpy())
        for cls_id in cls_ids:
            result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
        result_info_parts = []
        for idx_cls, cls_num in enumerate(result_nums):
            # only show detected classes
            if cls_num > 0:
                result_info_parts.append("{}:{}".format(result_names[idx_cls], cls_num))
        result_info = ", ".join(result_info_parts) if result_info_parts else "未检测到目标"
        self.img_num_label.setText("当前检测结果：{}".format(result_info))
        QMessageBox.information(self, "检测成功","检测成功")

    def open_cam(self):
        """打开摄像头上传"""
        self.webcam_detection_btn.setEnabled(False)    # 将打开摄像头的按钮设置为false，防止用户误触
        self.mp4_detection_btn.setEnabled(False)       # 将打开mp4文件的按钮设置为false，防止用户误触
        self.vid_stop_btn.setEnabled(True)             # 将关闭按钮打开，用户可以随时点击关闭按钮关闭实时的检测任务
        # self.vid_source = int(self.init_vid_id)        # 重新初始化摄像头
        if str(self.vid_source).isdigit():
            self.vid_source = int(self.vid_source)
        self.webcam = True                             # 将实时摄像头设置为true
        print(f"当前实时源：{self.vid_source}")
        self.cap = cv2.VideoCapture(self.vid_source)   # 初始化摄像头的对象
        th = threading.Thread(target=self.detect_vid)  # 初始化视频检测线程
        th.start()                                     # 启动线程进行检测

    def open_mp4(self):
        """打开mp4文件上传"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            # 和上面open_cam的方法类似，只是在open_cam的基础上将摄像头的源改为mp4的文件
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            self.vid_source = fileName
            self.webcam = False
            self.cap = cv2.VideoCapture(self.vid_source)
            th = threading.Thread(target=self.detect_vid)
            th.start()

    # 视频检测主函数
    def detect_vid(self):
        """检测视频文件，这里的视频文件包含了mp4格式的视频文件和摄像头形式的视频文件"""
        # model = self.model
        vid_i = 0
        track_history = defaultdict(lambda: [])
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 0:
            frame_width = 640
        if frame_height <= 0:
            frame_height = 480
        time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
        video_save_path = "record/vid/{}.mp4".format(time_re)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (frame_width, frame_height))
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            if success:
                # 如果是检测，也就是没有开检测器的话，就按照正常的检测流程走，如果此时开启了追踪，则应该进入追踪的分支按照追踪走
                if self.config_track_value.currentText() == "不开启追踪":

                    results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, save_txt=self.save_txt, save_conf=self.save_conf, save_crop=self.save_crop)
                    # 这个位置需要添加一个追踪的功能
                    result = results[0]
                    img_array = result.plot()
                    # 检测 展示然后保存对应的图像结果
                    im0 = img_array
                    im_record = copy.deepcopy(im0)
                    if self.video_writer is not None:
                        self.video_writer.write(im_record)
                    resize_scale = self.output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result_vid.jpg", im0)
                    self.set_scaled_pixmap(self.vid_img, "images/tmp/single_result_vid.jpg")
                    result_names = result.names
                    result_nums = [0 for i in range(0, len(result_names))]
                    cls_ids = list(result.boxes.cls.cpu().numpy())
                    for cls_id in cls_ids:
                        result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
                    result_info_parts = []
                    for idx_cls, cls_num in enumerate(result_nums):
                        if cls_num > 0:
                            result_info_parts.append("{}:{}".format(result_names[idx_cls], cls_num))
                    result_info = ", ".join(result_info_parts) if result_info_parts else "未检测到目标"
                    self.vid_num_label.setText("当前检测结果：{}".format(result_info))
                    vid_i = vid_i + 1
                else:
                    results = self.model.track(frame,  conf=self.conf_thres, iou=self.iou_thres, save_txt=self.save_txt,
                                         save_conf=self.save_conf, save_crop=self.save_crop, tracker=self.config_track_value.currentText(), persist=True)
                    # 这个位置需要添加一个追踪的功能
                    result = results[0]
                    img_array = result.plot()
                    # 尝试向image array上绘制检测的结果
                    try:
                        # Get the boxes and track IDs
                        boxes = results[0].boxes.xywh.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Plot the tracks
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # x, y center point
                            if len(track) > 30:  # retain 90 tracks for 90 frames
                                track.pop(0)

                            # Draw the tracking lines
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(img_array, [points], isClosed=False, color=(0, 0, 230),
                                          thickness=5)
                    except:
                        print("not got targets")
                    im0 = img_array
                    im_record = copy.deepcopy(im0)
                    if self.video_writer is not None:
                        self.video_writer.write(im_record)
                    resize_scale = self.output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result_vid.jpg", im0)
                    self.set_scaled_pixmap(self.vid_img, "images/tmp/single_result_vid.jpg")
                    result_names = result.names
                    result_nums = [0 for i in range(0, len(result_names))]
                    cls_ids = list(result.boxes.cls.cpu().numpy())
                    for cls_id in cls_ids:
                        result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
                    result_info_parts = []
                    for idx_cls, cls_num in enumerate(result_nums):
                        if cls_num > 0:
                            result_info_parts.append("{}:{}".format(result_names[idx_cls], cls_num))
                    result_info = ", ".join(result_info_parts) if result_info_parts else "未检测到目标"
                    self.vid_num_label.setText("当前检测结果：{}".format(result_info))
                    vid_i = vid_i + 1
            if cv2.waitKey(1) & self.stopEvent.is_set() == True:
                # 关闭并释放对应的视频资源
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                if self.cap is not None:
                    self.cap.release()
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                cv2.destroyAllWindows()
                self.reset_vid()
                break

    # 摄像头重置
    def reset_vid(self):
        """重置摄像头内容"""
        self.webcam_detection_btn.setEnabled(True)                      # 打开摄像头检测的按钮
        self.mp4_detection_btn.setEnabled(True)                         # 打开视频文件检测的按钮
        self.set_scaled_pixmap(self.vid_img, IMAGE_LEFT_INIT)                # 重新设置视频检测页面的初始化图像
        # self.vid_source = int(self.init_vid_id)                         # 重新设置源视频源
        self.webcam = True                                              # 重新将摄像头设置为true
        self.vid_num_label.setText("当前检测结果：{}".format("等待检测"))   # 重新设置视频检测页面的文字内容

    def close_vid(self):
        """关闭摄像头"""
        self.stopEvent.set()
        self.reset_vid()


    def check_record(self):
        """打开历史记录文件夹"""
        os.startfile(osp.join(os.path.abspath(os.path.dirname(__file__)), "record"))

    def save_config_change(self):
        #
        print("保存配置修改的结果")
        try:
            self.output_size = int(self.config_output_size_value.text())
            self.vid_source = str(self.config_vid_source_value.text())
            print(f"源地址:{self.vid_source}")
            # 添加对vid_source的初始化
            # self.cap =  cv2.VideoCapture(str(self.vid_source))
            self.vid_gap = int(self.config_vid_gap_value.text())
            self.conf_thres = float(self.config_conf_thres_value.text())
            self.iou_thres = float(self.config_iou_thres_value.text())
            ###

            # self.config_track_value.currentText()
            QMessageBox.information(self, "配置文件保存成功", "配置文件保存成功")
        except:
            QMessageBox.warning(self, "配置文件保存失败", "配置文件保存失败")



    def closeEvent(self, event):
        """用户退出事件"""
        reply = QMessageBox.question(self,
                                     '',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                # 退出之后一定要尝试释放摄像头资源，防止资源一直在线
                if self.cap is not None:
                    self.cap.release()
                    print("摄像头已释放")
            except:
                pass
            self.close()
            event.accept()
        else:
            event.ignore()
# 注册页面
class RegWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户注册")
        self.resize(280, 100)

        layout = QVBoxLayout()

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("请输入账号")

        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("请输入密码")
        self.pwd_input.setEchoMode(QLineEdit.Password)

        self.pwd_confirm = QLineEdit()
        self.pwd_confirm.setPlaceholderText("确认密码")
        self.pwd_confirm.setEchoMode(QLineEdit.Password)

        reg_btn = QPushButton("注册")
        reg_btn.clicked.connect(self.register)

        # layout.addWidget(QLabel("用户注册"))
        layout.addWidget(self.user_input)
        layout.addWidget(self.pwd_input)
        layout.addWidget(self.pwd_confirm)
        layout.addWidget(reg_btn)

        self.setLayout(layout)

    def register(self):
        username = self.user_input.text()
        password = self.pwd_input.text()
        confirm = self.pwd_confirm.text()

        if not username or not password:
            QMessageBox.warning(self, "错误", "用户名或密码不能为空")
            return

        if password != confirm:
            QMessageBox.warning(self, "错误", "两次密码不一致")
            return

        # 读取用户文件
        try:
            with open("users.json", "r", encoding="utf-8") as f:
                users = json.load(f)
        except:
            users = {}

        # 判断是否存在
        if username in users:
            QMessageBox.warning(self, "错误", "用户已存在")
            return

        # 保存
        users[username] = password

        with open("users.json", "w", encoding="utf-8") as f:
            json.dump(users, f, indent=4)

        QMessageBox.information(self, "成功", "注册成功！")
        self.close()

# 添加登录界面
class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        font_title = QFont("KaiTi", 16)
        self.setWindowTitle("登录界面")
        self.resize(800, 600)

        mid_widget = QWidget()
        window_layout = QFormLayout()
        self.user_name = QLineEdit()
        self.u_password = QLineEdit()
        window_layout.addRow("账号：", self.user_name)
        window_layout.addRow("密码：", self.u_password)
        self.user_name.setEchoMode(QLineEdit.Normal)
        self.u_password.setEchoMode(QLineEdit.Password)
        mid_widget.setLayout(window_layout)

        main_layout = QVBoxLayout()
        title_label = QLabel("智巡空域，精准识别")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            """
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #0B3D91;
                font-family: 'KaiTi';
            }
            """
        )

        logo_label = QLabel()
        pixmap = QPixmap("images/UI/picture1.png")
        pixmap = pixmap.scaled(520, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(title_label)
        main_layout.addWidget(logo_label)
        main_layout.addWidget(mid_widget)

        login_button = QPushButton("立即登陆")
        reg_button = QPushButton("注册账号")
        login_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0f6cbd;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #115ea3;
            }
            QPushButton:pressed {
                background-color: #0c4f8a;
            }
            """
        )
        reg_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e8f0f8;
                color: #17406d;
                border: 1px solid #c8d9eb;
                border-radius: 10px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #dbe9f6;
            }
            QPushButton:pressed {
                background-color: #cfdff0;
            }
            """
        )
        login_button.clicked.connect(self.login)
        reg_button.clicked.connect(self.open_register)

        main_layout.addWidget(login_button)
        main_layout.addWidget(reg_button)
        self.setLayout(main_layout)

        self.mainWindow = MainWindow()
        self.setFont(font_title)

    def open_register(self):
        self.regwindow = RegWindow()
        self.regwindow.show()


        # main_layout.addWidget(title_label)
        #
        # # a = QLabel(LOGIN_TITLE)
        # # a.setAlignment(Qt.AlignCenter)
        # # main_layout.addWidget(a)
        # # 图标
        # logo_label = QLabel()
        # pixmap = QPixmap("images/UI/picture1.png")
        # # 如果图片和 py 文件不在同一目录，就改成你的实际路径
        # pixmap = pixmap.scaled(520, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # logo_label.setPixmap(pixmap)
        # logo_label.setAlignment(Qt.AlignCenter)
        # main_layout.addWidget(logo_label)
        # # 输入框
        # main_layout.addWidget(mid_widget)
        #
        #
        #
        #
        # # self.setBa
        # # self.setObjectName("MainWindow")
        # # self.setStyleSheet("#MainWindow{background-color:rgb(236,99,97)}")
        #
        # login_button = QPushButton("立即登陆")
        # # reg_button = QPushButton("注册用户")
        # reg_button = QPushButton("注册账号")
        # reg_button.clicked.connect(self.open_register)
        # main_layout.addWidget(reg_button)
        # # reg_button.clicked.connect(self.reggg)
        # login_button.clicked.connect(self.login)
        #
        # # main_layout.addWidget(reg_button)
        # main_layout.addWidget(login_button)
        #
        # self.setLayout(main_layout)
        #
        # self.mainWindow = MainWindow()
        # self.setFont(font_title)
        # self.regwindow = RegWindow()

    # mainWindow.show()

    def login(self):
        username = self.user_name.text()
        password = self.u_password.text()

        # 1. 读取用户数据文件
        try:
            with open("users.json", "r", encoding="utf-8") as f:
                users = json.load(f)
        except:
            users = {}

        # 2. 判断用户名和密码
        if username in users and users[username] == password:
            QMessageBox.information(self, "成功", "登录成功！")
            self.mainWindow.show()
            self.close()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码错误")


# todo 添加模型参数的修改，以及添加对文件夹图像的加载
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = LoginWindow()
    mainWindow.show()
    sys.exit(app.exec())
