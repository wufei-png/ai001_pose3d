
import cv2
import numpy as np
import os
import time
from threading import Thread  # 创建线程的模块
import pyttsx3 
 
# aiff文件转换成mp3编码文件模块
from pydub import AudioSegment
import ntpath
import shutil
import torch.utils.data
from tqdm import tqdm
from joints_detectors.Alphapose.SPPE.src.main_fast_inference import *
from common.utils import calculate_area
from common.arguments import parse_args
from common.camera import *
from common.generators import UnchunkedGenerator
from common.model import *
from common.utils import Timer, evaluate, add_path
from joints_detectors.Alphapose.gene_npz import *
from joints_detectors.Alphapose.opt import opt
from joints_detectors.Alphapose.fn import getTime
from joints_detectors.Alphapose.dataloader import DetectionLoader, DetectionProcessor, DataWriter, Mscoco, VideoLoader
import matplotlib
from cal_angle import cal_angle
matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# def opecv_muti_pic(img1,img2):
#     # 图1
#     # imgs = np.hstack([img1,img2])
#     # 展示多个
#     cv2.imshow("mutil_pic1", img1)
#     cv2.imshow("mutil_pic2", img2)
#     #等待关闭
#     cv2.waitKey(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}#应该是左右两边把

add_path()

say_list=[]
def say_task(content):
    engine = pyttsx3.init()
 
# 输出文件格式
    # print('准备开始语音播报...')
# 设置要播报的Unicode字符串
    # if(engine._inLoop):
    #     print('说的话冲突啦')
    #     say_list.append('刚刚我在说别的东西，我刚要说的东西是'+content)
    # else:
    engine.say(content) 
    engine.runAndWait()

# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()

def check_content(say_list):
    while 1:
        engine = pyttsx3.init()
        if(len(say_list)==0):
            print('没啥事')
            pass
        else :
            if(engine._inLoop):
                print('还在说，别急')
                pass
            else :
                print('空闲了，你可以说了')
                engine.say(say_list[0]) 
                say_list=say_list[1:]
                engine.runAndWait()
        time.sleep(500)


def set_equal_aspect(ax, data):
    """
    Create white cubic bounding box to make sure that 3d axis is in equal aspect.
    :param ax: 3D axis
    :param data: shape of(frames, 3), generated from BVH using convert_bvh2dataset.py
    """
    X, Y, Z = data[..., 0], data[..., 1], data[..., 2]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

# def get_detector_2d(detector_name):
#     def get_alpha_pose():
#         from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
#         return alpha_pose

#     def get_hr_pose():
#         from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
#         return hr_pose

#     detector_map = {
#         'alpha_pose': get_alpha_pose,
#         'hr_pose': get_hr_pose,
#         # 'open_pose': open_pose
#     }

#     assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

#     return detector_map[detector_name]()

# def get_images():
#   imgs=[]
#   for i in range(1859):
#     imgs.append(str(i)+'.jpg')
#   print(imgs[len(imgs)-1])
#   print(imgs[0])
#   return imgs
class Skeleton:  #这是干嘛的
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]

# cap1 = cv2.VideoCapture('./kunkun_cut.mp4')
plt.ion()
def main():
    # p1 = Thread(target=check_content,args=(say_list,))
    # p1.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行
    img_init=False
    args = opt
    # args.vis_fast=True
    args.dataset = 'coco'
    args.fast_inference = True#true路径不对 bug
    args.save_img = False
    args.sp = True


        # 2.循环读取图片
    video_file='./outputs/Sitting3.mp4'
    args.video = video_file
    base_name = os.path.basename(args.video)
    video_name = base_name[:base_name.rfind('.')]
    args.outputpath = f'outputs/alpha_pose_{video_name}'
    if os.path.exists(args.outputpath):
        shutil.rmtree(f'{args.outputpath}/vis', ignore_errors=True)
    else:
        os.mkdir(args.outputpath)
    videofile = args.video
    mode = args.mode
    if not len(videofile):
        raise IOError('Error: must contain --video')
    # Load input video
    det_processor,im_names_desc=get_video_handle(video_file)
    print('Loading YOLO model..')
    print('torch.cuda.is_available()',torch.cuda.is_available())
    sys.stdout.flush()
    pose_dataset = Mscoco()
    if args.fast_inference:
        print('InferenNet_fast')
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        print('InferenNet_slow')
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    batchSize=80
    writer = DataWriter(args.save_video).start()
    print('Start pose estimation...')
    time0 = ckpt_time()

    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=True, dropout=0.25, channels=1024,dense=False)

    chk_filename='checkpoint\pretrained_h36m_detectron_coco.bin'
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint['model_pos'])
    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    print('INFO: Using causal convolutions')
    causal_shift = pad

    # fig = plt.figure(figsize=(10, 5))
    # ax_in = fig.add_subplot(1, 2, 1)
    # ax_in.get_xaxis().set_visible(False)
    # ax_in.get_yaxis().set_visible(False)
    # ax_in.set_axis_off()
    # ax_in.set_title('Input')
    # radius = 1.7

    # ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    # ax_3d.view_init(elev=15., azim=np.array(70., dtype=np.float32))
    # ax_3d.set_xlim3d([-radius / 2, radius / 2])
    # ax_3d.set_zlim3d([0, radius])
    # ax_3d.set_ylim3d([-radius / 2, radius / 2])
    # # ax_3d.set_aspect('equal')
    # ax_3d.set_xticklabels([])
    # ax_3d.set_yticklabels([])
    # ax_3d.set_zticklabels([])
    # ax_3d.dist = 12.5
    # ax_3d.set_title('Reconstrcution')  # , pad=35
    # tmp=[]
    # tmp.append(ax_3d)
    # ax_3d=tmp

    # image = ax_in.imshow(np.zeros((480, 864, 3)), aspect='equal')
    # points = ax_in.scatter(1,1, 5, color='red', edgecolors='white', zorder=10)
    if torch.cuda.is_available():
        print('cuda is ok!!!!!!!!!!!')
        model_pos = model_pos.cuda()
    leg_flag={'left':0,'right':0,'back':0}
    # 0 normal -1 蜷缩/前趴 1 伸长/后仰
    for i in im_names_desc:
        fps=0
        time0 = ckpt_time()
        # print('i',i)
        if (i==1):
            img_init=True
        # print('img_init',img_init)
        # ret2, frame2 = cap2.read()#替换
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2)=det_processor.read()
            # print('type(orig_img)',type(orig_img))
            # cv2.imshow("frame1", orig_img)
            # frame =  np.array(orig_img)
            # cv2.imshow("frame2", frame)
            key = cv2.waitKey(25)
            if key == 27:  # 按键esc
                return
            if orig_img is None:
                # print(f'{i}-th image read None: handle_video')
                break
            if boxes is None or boxes.nelement() == 0:
                # print('这就解释的通了1')
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            # print('datalen',datalen) #1
            if datalen % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)

            hm = hm.cpu().data
            # print('这就解释的通了')
            # print(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            while writer.running():
                pass
            result = writer.get_latest_result()
            result_list=[]
            result_list.append(result)
            # print('len(result_list)',len(result_list))
            result_list=get_finnal_kpt(result_list)
            # print('result_list shape',result_list.shape)
            # print('result:',result)
            # print('np.array(result).shape',np.array(result).shape)
            keypoint = normalize_screen_coordinates(result_list[..., :2], w=1000, h=1002)
            ckpt, time124 = ckpt_time(time0)
            fps=round(1/ckpt,2)
            print('fps:',fps)
            cv2.imshow("frame1", orig_img)
            # # print('keypoint.shape',keypoint.shape)
            # gen = UnchunkedGenerator(None, None, [keypoint],
            #                  pad=pad, causal_shift=causal_shift, augment=True,
            #                  kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            # prediction = evaluate(gen, model_pos, return_predictions=True)
           
            # # print('prediction[00]',prediction[0])
            # rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
            
            # prediction = camera_to_world(prediction, R=rot, t=0)
            # # print('prediction[01]',prediction[0])
            # # print('prediction',prediction)
            # # print('prediction.shape',prediction.shape)
            # # prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            # # print('prediction[02]',prediction[0])
            # # anim_output = {'Reconstruction': prediction}
            # # print('anim_output',prediction[0])
            # input_keypoints = image_coordinates(keypoint[..., :2], w=1000, h=1002)
            # # print('input_keypoints',input_keypoints.shape)
            # input_keypoints=input_keypoints[0]
            # for i in range(17):
            #     # print('input_keypoints[i][0]',input_keypoints[i][0])
            #     # print('input_keypoints[i][0] type ',type(input_keypoints[i][0]))
            #     #鼻子 左眼 右眼 左耳 右耳 左肩 右肩 左 右肘 左 右腕 左 右臀
            #     # 左右膝 左右脚踝
            #     cv2.circle(orig_img,(int(input_keypoints[i][0]),int(input_keypoints[i][1])),5,(0, 0, 255),-1)
            #     #语音提示：
            # # print('prediction shape',prediction.shape)
            # # prediction shape (1, 17, 3)
            # # prediction type <class 'tuple'>
            # # print('prediction type',type(prediction.shape))
            # # prediction=prediction[0]
            # prediction = list(prediction)
            # prediction = prediction[0]
            # # print('prediction 1',prediction)
            # left_leg=[]
            # right_leg=[]
            # back=[]
            # left_leg.append(prediction[4])
            # left_leg.append(prediction[5])
            # left_leg.append(prediction[6])

            # right_leg.append(prediction[1])
            # right_leg.append(prediction[2])
            # right_leg.append(prediction[3])
            # # c=[]
            # # for i in range(len(prediction[2])):
            # #     c.append((prediction[2][i]+prediction[5][i])/2)
            #     # print(c)
            # back_vertice=[0,0,1]
            # back_vertice[:2]=prediction[0][:2]

            # back.append(back_vertice)
            # back.append(prediction[0])
            # back.append(prediction[8])
            # return_info=check_angle(left_leg,right_leg,back,leg_flag)
            # #大于零
            # # tips=check_pose(prediction)
            # ckpt, time124 = ckpt_time(time0)
            # fps=round(1/ckpt,2)
            # return_info='fps:'+str(fps)+'\n'+return_info
            # orig_img=cv2AddChineseText(orig_img,return_info, (50, 50),(0, 255, 0), 30)

            # cv2.imshow("frame1", orig_img)
            # if(img_init==False):
            #     image = ax_in.imshow(frame, aspect='equal')
            #     points = ax_in.scatter(*input_keypoints[0].T, 5, color='red', edgecolors='white', zorder=10)
            # else:
            #     print('执行这一步了')
            #     image.set_data(frame)
            #     points.set_offsets(*input_keypoints[0].T)
            # plt.pause(0.01)
            # show_3dimage(input_keypoints,anim_output,Skeleton(),np.array(70., dtype=np.float32),viewport=(1000, 1002),frame=frame,ax_3d=ax_3d,ax_in=ax_in,fig=fig,image=image,points=points,img_init=img_init)
    writer.stop()
    ckpt, time124 = ckpt_time(time0)
    print('-------------- 334帧 spends {:.2f} seconds'.format(ckpt))
    plt.ioff()
def check_angle(left_leg,right_leg,back,leg_flag):
    tips=''
    say_it=False
    left_angle=cal_angle(left_leg[0],left_leg[1],left_leg[2])-30#这个角度计算稍微有点大
    right_angle=cal_angle(right_leg[0],right_leg[1],right_leg[2])-30
    back_angle=cal_angle(back[0],back[1],back[2])-15
    tips=tips+'左腿角度：'+str(left_angle)[:5]+'°'+'\n'
    tips=tips+'右腿角度：'+str(right_angle)[:5]+'°'+'\n'
    tips=tips+'背部角度约为：'+str(back_angle)[:5]+'°'
    return_info=tips
    tips=''
    if(left_angle<=45 and (leg_flag['left']!=-1)):
        leg_flag['left']=-1
        say_it=True
        tips=tips+'左腿角度小于45度，长期可能静脉曲张'

    if(right_angle<=45 and (leg_flag['right']!=-1)):
        leg_flag['right']=-1
        say_it=True
        tips=tips+'右腿角度小于45度，长期可能静脉曲张'

    if(left_angle>45 and left_angle<120 and (leg_flag['left']!=0)):
        leg_flag['left']=0
        say_it=True
        tips=tips+'左腿角度很不错'

    if(right_angle>45 and right_angle<120 and (leg_flag['right']!=0)):
        leg_flag['right']=0
        say_it=True
        tips=tips+'右腿角度很不错'

    if(left_angle>120 and (leg_flag['left']!=1)):
        leg_flag['left']=1
        say_it=True
        tips=tips+'左腿角度大于120度，处于放松状态，记得多活动'

    if(right_angle>120 and (leg_flag['right']!=1)):
        leg_flag['right']=1
        say_it=True
        tips=tips+'右腿角度大于120度，处于放松状态，记得多活动'

    if(back_angle>30 and (leg_flag['back']!=1)):
        leg_flag['back']=1
        say_it=True
        tips=tips+'背前倾或者后仰过多，请及时调整'

    if(back_angle<30  and (leg_flag['back']!=0)):
        leg_flag['back']=0
        say_it=True
        tips=tips+'背部角度很不错'

    if(say_it):
        p1 = Thread(target=say_task,args=(tips,))
        p1.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行
    return return_info

def show_3dimage(keypoints,poses,skeleton,azim,viewport,frame,ax_3d,ax_in,fig,image,points,img_init):
    fig.clf()
    trajectories = []
    lines_3d=[]
    for index, (title, data) in enumerate(poses.items()):
        # trajectories.append(data[:, 0, [0, 1]])
        pass
    poses = list(poses.values())
    parents = skeleton.parents()
    if(img_init==False):
        image = ax_in.imshow(frame, aspect='equal')
        # points = ax_in.scatter(*keypoints[0].T, 5, color='red', edgecolors='white', zorder=10)
    else:
        # print('执行这一步了')
        image.set_data(frame)
        # points.set_offsets(keypoints)
    plt.show()
    plt.pause(0.01)
    # print('*keypoints[0].T')


def get_finnal_kpt(final_result):
    # ============ Changing ++++++++++
    kpts = []
    no_person = []
    for i in range(len(final_result)):
        if not final_result[i]['result']:  # No people
            no_person.append(i)
            kpts.append(None)
            continue

        kpt = max(final_result[i]['result'],
                  key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']

        kpts.append(kpt.data.numpy())

        for n in no_person:
            kpts[n] = kpts[-1]
        no_person.clear()

    for n in no_person:
        kpts[n] = kpts[-1] if kpts[-1] else kpts[n-1]

    # ============ Changing End ++++++++++
    kpts = np.array(kpts).astype(np.float32)
    # print('kpts shape',kpts.shape)
    return kpts

    # cv2.imshow("frame2", frame2)


    #     # 退出播放
    #     key = cv2.waitKey(25)
    #     if key == 27:  # 按键esc
    #         break

    # 3.释放资源

    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
