import sys
from PyQt5.QtWidgets import QFileDialog,QApplication, QWidget, QMainWindow, QLabel,QPushButton,QSizePolicy
from PyQt5.QtGui import QIcon,QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import os
from predict_name import face_detect




class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Image Viewer'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 760
        self.widget_width=40
        self.widget_height=30
        self.label_width=800
        self.label_height=720
        self.initUI()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.display_backward()
        elif event.key() == Qt.Key_Right:
            self.display_forward()
        else:
            pass
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.count=0
        self.total=0
        
        self.button1=QPushButton(self)
        self.button1.resize(self.widget_width,self.widget_height)
        self.button1.move(int(self.width/2)-20,720)
        self.button1.clicked.connect(self.open_files)
        # self.button1.setIcon(QIcon('play.png'))
        self.button1.setIcon(QIcon(('play.png')))
        self.button1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.button2=QPushButton(self)
        self.button2.resize(self.widget_width,self.widget_height)
        self.button2.move(int(self.width/2)-80,720)
        self.button2.clicked.connect(self.display_backward)
        self.button2.setIcon(QIcon(('back_icon.png')))
        self.button2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        
        self.button3=QPushButton(self)
        self.button3.resize(self.widget_width,self.widget_height)
        self.button3.move(int(self.width/2)+40,720)
        self.button3.clicked.connect(self.display_forward)
        self.button3.setIcon(QIcon(('forward_icon.png')))
        self.button3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.button4=QPushButton(self)
        self.button4.resize(self.widget_width,self.widget_height)
        self.button4.move(int(self.width/2)+180,720)
        self.button4.clicked.connect(self.on_exit)
        self.button4.setIcon(QIcon(('power_off.png')))
        self.button4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        self.label=QLabel(self)
        self.label.move(10,10)
        self.label.resize(self.label_width,self.label_height)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.show()

    def open_files(self):
        dialog=QFileDialog()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #dialog.setNameFilters(["*png"])
        self.files, _ = dialog.getOpenFileNames(self,"Open files", "","Image files (*.png *.webp *.jpg *.jpeg)",options=options)

        if self.files is not None:
            self.total=len(self.files)
            self.count=0
        self.display_image()

    def display_backward(self):
        if self.count ==0:
            self.count=self.total - 1
        else:
            self.count -=1
        self.display_image()

    def display_forward(self):
        if self.count == self.total -1:
            self.count =0
        else:
            self.count +=1

        self.display_image()

    def display_image(self):
        print(self.files[self.count])
        title_name = predict_name(self.files[self.count])
        self.setWindowTitle(title_name)
        if self.total!=0:
            pixmap = QPixmap(self.files[self.count])
            w=pixmap.size().width()
            h=pixmap.size().height() 
            # print("original:{},{}".format(w,h))
            if w>=self.label_width or h>=self.label_height:
                pixmap_resize=pixmap.scaled(self.label_width, self.label_height, QtCore.Qt.KeepAspectRatio,transformMode=QtCore.Qt.SmoothTransformation)
                w=pixmap_resize.size().width()
                h=pixmap_resize.size().height()
                # print("modified:{},{}".format(w,h))
            else:
                pixmap_resize=pixmap.copy()

            self.label.setPixmap(pixmap_resize)
            self.label.move(int(self.width/2)-int(0.5*w),0)

    def on_exit(self):
        sys.exit(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
