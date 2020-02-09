import sys
from LoginWindow import LoginWindow
from MainWindow import MainWindow
from Loginerror import Loginerror
from Alternative import Alternative
from Reflation import Reflation
from Photo import PhotoCatch
import time
import cv2
import numpy as np
import Hotkey
from MyModel import runTracker
from PyQt5.QtWidgets import QApplication,QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

runtracher = runTracker()

class Main(QMainWindow,MainWindow):
    def __init__(self):
        super(Main,self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.close_func)
        self.timer_camera = QTimer()
        self.timer_camera.start(300)
        self.timer_init()
        self.flag = True
        if runtracher.isOpenVideoCap() != True:
            QMessageBox.information(runtracher,  # 使用infomation信息框
                                            "错误",
                                            "打开摄像头失败，请检查配置文件中的链接是否有效",
                                            QMessageBox.Ok)
            return
    def timer_init(self):
        self.timer_camera.timeout.connect(self.reflash_func)

    def reflash_func(self):
        t1 = time.time()
        if self.flag == True:
            full_img, local_img, hf_img = runtracher.CreateImg()

        # full
            show = cv2.cvtColor(full_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(showImage))

        # local
            show1 = cv2.cvtColor(local_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            showImage = QImage(show1.data, show1.shape[1], show1.shape[0], QImage.Format_RGB888)
            self.label_4.setPixmap(QPixmap.fromImage(showImage))

            dict, num = runtracher.PredictImg(local_img)
            if dict == "":
                self.label_2.setText(u'识别失败')
            # print('G P fail')
            else:
                self.label_2.setText(u'识别的手势为：' + dict)

            self.label_3.setText('取图时间： %.3f' % ((time.time() - t1) ))

    def close_func(self):
        self.close()
        self.timer_camera.stop()
        self.label.setPixmap(QPixmap(""))
        self.label_4.setPixmap(QPixmap(""))
        self.flag = False
        self.alternative = Alternative()
        self.alternative.show()

class Login(QMainWindow,LoginWindow):
    def __init__(self):
        super(Login,self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.check)
    def check(self):
        useraccount = self.lineEdit.text()
        userpassword = self.lineEdit_2.text()
        if useraccount == "hellenesis" and userpassword =="123456":
            self.close()
            self.alternative = Alternative()
            self.alternative.show()
        else:
            self.close()
            self.loginerror = Loginerror()
            self.loginerror.show()

class Loginerror(QMainWindow,Loginerror):
    def __init__(self):
        super(Loginerror,self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.back)
    def back(self):
        self.close()
        self.login = Login()
        self.login.show()

class Alternative(QMainWindow,Alternative):
    def __init__(self,a="",b="",c=""):
        super(Alternative,self).__init__()
        self.setupUi(self)
        self.value = a
        self.value2 = b
        self.dict = c
        self.pushButton.clicked.connect(self.toRealtime)
        self.pushButton_2.clicked.connect(self.toPhoto)
        self.pushButton_3.clicked.connect(self.toBack)
        self.pushButton_4.clicked.connect(self.toReflation)
    def toRealtime(self):
        self.close()
        self.main = Main()
        self.main.show()
    def toPhoto(self):
        self.close()
        self.photo = Photo()
        self.photo.show()
    def toBack(self):
        self.close()
        self.login = Login()
        self.login.show()
    def toReflation(self):
        self.close()
        self.reflation = Reflation(self.value,self.value2,self.dict)
        self.reflation.show()

class Reflation(QMainWindow,Reflation):
    def __init__(self,value="",value2="",dict=""):
        super(Reflation,self).__init__()
        self.setupUi(self)
        self.timer_camera = QTimer()
        self.timer_camera.start(300)
        self.timer_init()
        self.value = value
        self.value2 =value2
        self.dict = dict
        print(value+value2+dict)
        if runtracher.isOpenVideoCap() != True:
            QMessageBox.information(runtracher,  # 使用infomation信息框
                                    "错误",
                                    "打开摄像头失败，请检查配置文件中的链接是否有效",
                                    QMessageBox.Ok)
            return

    def timer_init(self):
        self.timer_camera.timeout.connect(self.reflash_func)

    def reflash_func(self):
        t1 = time.time()
        full_img, local_img, hf_img = runtracher.CreateImg()

        show1 = cv2.cvtColor(local_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        showImage = QImage(show1.data, show1.shape[1], show1.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))
        dict, num = runtracher.PredictImg(local_img)

        if(dict == self.dict):
            Hotkey.keyattach(self.value,self.value2)

##############################################################################################################

class Photo(QMainWindow,PhotoCatch):
    def __init__(self):
        super(Photo,self).__init__()
        self.setupUi(self)
        self.timer_camera = QTimer()
        self.timer_init()
        self.Back.clicked.connect(self.toBack)
        self.catch_2.clicked.connect(self.finaldict)
        self.reaction.clicked.connect(self.react)
        self.realized.clicked.connect(self.reborn)
        self.final=""
    def toBack(self):
        self.close()
        self.timer_camera.stop()
        self.photoget.setPixmap(QPixmap(""))
        self.photo.setPixmap(QPixmap(""))
        a,b,c = self.react()
        self.alternative = Alternative(a,b,c)
        self.alternative.show()

    def timer_init(self):
        self.timer_camera.start(10)
        self.timer_camera.timeout.connect(self.catch)
    def catch(self):
        full_img, local_img, hf_img = runtracher.CreateImg()
        show = cv2.cvtColor(full_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.photo.setPixmap(QPixmap.fromImage(showImage))

        full_img, local_img, hf_img = runtracher.CreateImg()
        show1 = cv2.cvtColor(local_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        showImage = QImage(show1.data, show1.shape[1], show1.shape[0], QImage.Format_RGB888)
        self.photoget.setPixmap(QPixmap.fromImage(showImage))

        dict,num = runtracher.PredictImg(local_img)
        if dict == "":
            self.TureGesture.setText(u'识别未成功')
        else:
            self.final = dict
            self.TureGesture.setText(u'识别成功手势:'+dict)
    def finaldict(self):
        self.photoget.setPixmap(QPixmap(""))
        self.photo.setPixmap(QPixmap(""))
        self.timer_camera.stop()
        self.TureGesture.setText("识别最终手势："+self.final)
        self.photo.setText("手势识别停止")
        self.photoget.setText("手势获取停止")
    def react(self):
        hotkeyvalue1 = self.hotkey.text()
        hotkeyvalue2 = self.hotkey2.text()
        hotkeyconnect = self.final
        return hotkeyvalue1,hotkeyvalue2,hotkeyconnect
    def reborn(self):
        self.timer_init()