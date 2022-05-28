
if __name__ == '__main__':
  from threading import Thread  # 创建线程的模块
  from multiprocessing import  Process
  from pose_3d import pose_3d
  from hand_write import hand_write
  print('执行了几次')
  # p1 = Process(target=hand_write)
  p1 = Process(target=pose_3d)
  p1.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行
  p2 = Process(target=hand_write)
  p2.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行