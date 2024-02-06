import cv2
import mediapipe as mp
import numpy as np
# 导入音量控制模块
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
# ll = [4, 8, 12, 16, 20]
def get_str_guester(up_fingers):
    if len(up_fingers) == 1:
        str_guester = "1"
        if up_fingers[0] == 4:
            str_guester = "Good"
        elif up_fingers[0] == 20:
            str_guester = "Bad"
    elif len(up_fingers) == 2:
        str_guester = "2"
    elif len(up_fingers) == 3:
        str_guester = "3"
    elif len(up_fingers) == 4:
        str_guester = "4"
    elif len(up_fingers) == 5:
        str_guester = "5"
    else:
        str_guester = " "
    return str_guester

def mouse(up_fingers):
    if len(up_fingers) == 2 and up_fingers[0]==4 and up_fingers[1]==8:
        return up_fingers
    else:
        return

if __name__ == '__main__':

    # 获取音响设备
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()  # 静音
    # volume.GetMasterVolumeLevel()  # 获取主音量级
    volRange = volume.GetVolumeRange()  # 音量范围(-96.0, 0.0)
    # print(volRange)
    # 设置最值音量
    minVol = volRange[0]  # 元素：-96.0
    maxVol = volRange[1]  # 元素：0
    # 打开摄像机Camera0
    cap = cv2.VideoCapture(0)
    Rlength = []
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    # 创建检测手部关键点和关键点之间连线的方法
    mpDraw = mp.solutions.drawing_utils
    while True:
        # 读取摄像机的视频图像,会返回两个值：Ture或False 和 帧
        success, img = cap.read()
        # 图像的长、宽、通道
        image_height,image_width, image_cross = img.shape
        # 检测到图片后，我们便可以直接使用图片检测的步骤，进行模型的检测
        # 因为在opencv中使用的色彩格式为BGR，而mediapipe能识别的为RGB，所以要先用img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 将BGR转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.flip
        # 识别图像中的手势，并返回结果
        results = hands.process(imgRGB)
        # results.multi_hand_landmarks为None时进行for循环会报错，所以要先判断
        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms,
                                      mpHands.HAND_CONNECTIONS,
                                      mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                      mp.solutions.drawing_styles.get_default_hand_connections_style()
                                      )

            # 采集所有关键点的坐标
            list_lms = []
            RlmList = []
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17]
            # 手部关键点的坐标被归一化到图像的宽度和高度范围内。通过将相对坐标乘以图像的宽度和高度，将它们转换为与图像尺寸相对应的比例值
            for i in range(21):
                pos_x = handLms.landmark[i].x * image_width
                pos_y = handLms.landmark[i].y * image_height
                list_lms.append([int(pos_x), int(pos_y)])
            # 构造凸包点
            list_lms = np.array(list_lms, dtype=np.int32)
            # Rlist_lms = np.array(Rlist_lms, dtype=np.int32)
            # 使用凸包算法计算指定关键点索引处的凸包
            hull = cv2.convexHull(list_lms[hull_index, :])
            # 绘制凸包 True 表示绘制闭合的多边形，连接凸包的起点和终点
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)
            # 查找外部的点数
            ll = [4, 8, 12, 16, 20]
            up_fingers = []
            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    up_fingers.append(i)
            # print(len(up_fingers))
            print(up_fingers)
            # print(list_lms)
            # print(np.shape(list_lms))
            # print(results.multi_handedness[0].classification[0])
            # 为镜像，所以我的左手labe为Right
            if results.multi_handedness[0].classification[0].label == "Right":
                str_guester = get_str_guester(up_fingers)
                cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
            elif results.multi_handedness[0].classification[0].label == "Left":
                try:
                    if mouse(up_fingers):
                        # 向上取整，得到手指坐标的整数
                        x1=math.ceil(handLms.landmark[up_fingers[0]].x * image_width)
                        y1=math.ceil(handLms.landmark[up_fingers[0]].y * image_height)
                        x2=math.ceil(handLms.landmark[up_fingers[1]].x * image_width)
                        y2=math.ceil(handLms.landmark[up_fingers[1]].y * image_height)
                        # 保存坐标点
                        RlmList.append([[x1, y1], [x2, y2]])
                        # 基于长度控制音量
                        # 计算线段之间的长度，勾股定理计算平方和再开根
                        length = int(math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))
                        Rlength.append(length)
                        # print(length)
                        print(max(Rlength), min(Rlength))
                        # 在拇指和食指中间画一条线段，img画板，起点和终点坐标，颜色，线条宽度
                        # 线段长度最大280，最小10，转换到音量范围，最小-96，最大0
                        # 将线段长度变量length从[10,280]转变成[-96,0]
                        # 选取自己舒服的范围即可
                        # vol = np.interp(length, [10, 250], [minVol, maxVol])
                        vol = np.interp(length, [30, 280], [minVol, maxVol])
                        # print('vol:', vol, 'length:', length)
                        # 设置电脑主音量
                        volume.SetMasterVolumeLevel(vol, None)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except:
                    pass

            else:
                cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)

            for i in ll:
                pos_x = handLms.landmark[i].x * image_width
                pos_y = handLms.landmark[i].y * image_height
                # 画点
                cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
        # 显示视频图像
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

