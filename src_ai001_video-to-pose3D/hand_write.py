# def get_images():
#   imgs=[]
#   for i in range(1859):
#     imgs.append(str(i)+'.jpg')
#   print(imgs[len(imgs)-1])
#   print(imgs[0])

#   return imgs
import sys
import cv2
import numpy as np
import os
import time
# from test import pose_3d
from threading import Thread  # 创建线程的模块
# p1 = Thread(target=pose_3d)
# p1.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行
# sys.path.append('com_ocr')
# get_images()
from com_ocr.com_ocr import predict
from FingerCounter.com_finger import counter
#输入要识别的图片名称
format_file=0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def hand_write():
  save_file=False
  cap1 = cv2.VideoCapture('./outputs/pdf.mp4')
  cap2 = cv2.VideoCapture('./outputs/write.mp4')
  # print(666)
  # print(cap1.isOpened() )
  # print(cap2.isOpened() )
  index=0
  while(cap1.isOpened() and cap2.isOpened()):
      ret1, frame1 = cap1.read()
      ret2, frame2 = cap2.read()
      if(ret1 and ret2):#取最小的长度
          img,sign=counter(frame1)
          # print('img',img)
          # print('sign',sign)
          cv2.imshow("fram1",img)
          key = cv2.waitKey(25)
          if key == 27:  # 按键esc
            return
          # time.sleep (40)
          cv2.imwrite('./imgs_result/'+str(index)+'.png', img)
          index=index+1
          if(sign!=1 and sign!=2 and sign!=3):
            continue
          else:
            if(sign==1 or sign==2):
              format_file=sign
              continue
            if(sign==3 and save_file==False):
              save_file=True
              p1 = Thread(target=predict,args=(frame2,format_file,))
              p1.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行
              # predict(frame2,format_file)
              # break#整个程序结束

      else:
        break
#主线程会等待所有的子线程执行结束再结束
  cap1.release()
  cap2.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    hand_write()


