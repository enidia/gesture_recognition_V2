# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\Alternative.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Alternative(object):
    def setupUi(self, Alternative):
        Alternative.setObjectName("Alternative")
        Alternative.resize(400, 400)
        Alternative.setMinimumSize(QtCore.QSize(400, 400))
        Alternative.setMaximumSize(QtCore.QSize(400, 400))
        self.centralwidget = QtWidgets.QWidget(Alternative)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(200, 50))
        self.pushButton_2.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMinimumSize(QtCore.QSize(200, 50))
        self.pushButton.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setMinimumSize(QtCore.QSize(200, 50))
        self.pushButton_3.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 2, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setMinimumSize(QtCore.QSize(200, 50))
        self.pushButton_4.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 3, 0, 1, 1)
        Alternative.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Alternative)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 400, 26))
        self.menubar.setObjectName("menubar")
        Alternative.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Alternative)
        self.statusbar.setObjectName("statusbar")
        Alternative.setStatusBar(self.statusbar)

        self.retranslateUi(Alternative)
        QtCore.QMetaObject.connectSlotsByName(Alternative)

    def retranslateUi(self, Alternative):
        _translate = QtCore.QCoreApplication.translate
        Alternative.setWindowTitle(_translate("Alternative", "Alternative"))
        self.pushButton_2.setText(_translate("Alternative", "拍照识别"))
        self.pushButton.setText(_translate("Alternative", "实时识别"))
        self.pushButton_3.setText(_translate("Alternative", "退回登录"))
        self.pushButton_4.setText(_translate("Alternative", "效应界面"))

