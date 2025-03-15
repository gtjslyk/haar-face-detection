import neg_generator,cv2
from cascade_classifier import *
import random

import numpy as np



import argparse


import cProfile

parser = argparse.ArgumentParser(  
        description='')  
parser.add_argument(  
    '-s', '--stage', default = 1, metavar='int', type=int,  
    help='stage to test') 
parser.add_argument(  
    '-d', '--divide', default = 4, metavar='int', type=int,  
    help='divide scale') 
parser.add_argument(  
    '-p', '--path', default = 'a.jpg', metavar='str', type=str,  
    help='path') 
parser.add_argument(  
    '-c', '--cap', default = True, metavar='bool', type=lambda x : True if x == 't' else False,  
    help='capture') 

args = parser.parse_args()
stage = args.stage
path = args.path
divide = args.divide
cap = args.cap


def sliding_window_viola_jones(image, min_window_size, scale_factor=1.2, step_size=10):
    """
    实现Viola-Jones算法中的滑动窗口（窗口逐渐放大）
    :param image: 输入的图像
    :param min_window_size: 最小窗口大小，通常为一个固定大小 (宽, 高)
    :param scale_factor: 每次放大的比例因子，通常大于1
    :param step_size: 每次滑动的步长
    :return: 生成器，返回当前窗口的图像和窗口的位置
    """
    # 获取图像的高度和宽度
    h, w = image.shape[:2]

    # 初始化窗口大小
    window_width = min_window_size

    # 持续增加窗口大小直到超出图像边界
    while window_width <= h and window_width <= w:
        # 在当前窗口大小下进行滑动窗口操作
        for y in range(0, h - window_width + 1, step_size):
            for x in range(0, w - window_width + 1, step_size):
                window = image[y:y + window_width, x:x + window_width]
                cc+=1
                yield window, (x, y)

        # 按照scale_factor放大窗口
        window_width = int(window_width * scale_factor)


def draw_squares(image, squares):
    """
    在图像中绘制正方形
    :param image: 输入的图像
    :param squares: 包含正方形的列表，每个正方形由一个元组 (x, y, width) 表示
    :return: 绘制完正方形的图像
    """
    # 复制原图，避免修改原始图像
    img_copy = image.copy()

    for (x, y, width) in squares:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (0, 255, 255)
        # 绘制正方形，(x, y) 为左上角，(x + width, y + width) 为右下角
        cv2.rectangle(img_copy, (x, y), (x + width, y + width), color, 1)  # 绿色，线宽为2

    return img_copy



def process(aa):

    min_window_size = 24
    max_window_size = 40
    scale_factor=1.2
    image = aa
    step_factor = 2

    # 获取图像的高度和宽度
    h, w = image.shape[:2]

    # 初始化窗口大小
    window_width = min_window_size
    out = []
    cc = 0
    # 持续增加窗口大小直到超出图像边界
    while window_width <= h and window_width <= w and window_width <= max_window_size:
        windows = []
        coords = []
        step_size = int(scale_factor * step_factor)
        # 在当前窗口大小下进行滑动窗口操作
        for y in range(0, h - window_width + 1, step_size):
            for x in range(0, w - window_width + 1, step_size):
                cc += 1
                window = image[y:y + window_width, x:x + window_width]
                # window = cv2.resize(window, (24, 24))
                windows.append(window)
                coords.append((x, y))
        """"""
        windows = np.array(windows)
        profiler = cProfile.Profile()
        profiler.enable()
        y_pred = c.predict(windows)
        profiler.disable()
        profiler.print_stats(sort='cumulative')
        windows_pred = windows[y_pred == True]
        # for win in windows_pred:
        #     cv2.namedWindow('a', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('a', 500, 500)
        #     cv2.imshow('a', win)
        #     cv2.waitKey(0)
        coords_pred = [coords[i] for i in range(len(coords)) if y_pred[i]]
        part_out = [(coord[0], coord[1], window.shape[0]) for window, coord in zip(windows_pred, coords_pred)]
        for window, coord in zip(windows_pred, coords_pred):
            pass
        out += part_out

        """"""
        # 按照scale_factor放大窗口
        window_width = int(window_width * scale_factor)
        

    
    print(f'windows: {len(out)} / {cc}')
    
    out = [(o[0], o[1], o[2], o[2]) for o in out]

    # out = np.array(out)
    # scores = np.ones((out.shape[0]))

    # indices = cv2.dnn.NMSBoxes(out, scores, 0, 0.1)
    # out = out[indices]
    out = [(o[0], o[1], o[2]) for o in out]

    return out

c = cascade_classifier()

for s in range(1, stage + 1):
    tmp = adaboost_classifier(classifier_name=f'stage_{s}')
    tmp.load_classifier(os.path.join('output',f'stage_{s}_classifier.pkl'))
    c.add_adaboost_calssifier(tmp)






if cap:
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    scale =  divide


    while True:
        # 读取一帧
        ret, frame = cap.read()
        # 如果读取失败，退出循环
        if not ret:
            print("无法读取帧")
            break

        # 在这里添加你对帧的处理代码
        # 例如：将帧转换为灰度图像
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.resize(processed_frame, (int(processed_frame.shape[1] / scale), int(processed_frame.shape[0] / scale)))
        print(processed_frame.shape)
        rects = process(processed_frame)
        # rects = [(x*scale, y*scale, w*scale) for x, y, w in rects]
        # print(rects)
        # processed_frame = draw_squares(frame, rects)

        # 显示原始帧和处理后的帧
        # cv2.imshow('Original Frame', frame)
        cv2.imshow('Processed Frame', processed_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if cap:
    exit()



# exit()

aa = cv2.imread(path)
aa = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY)
# aa = cv2.resize(aa, (500, 500))
aa = cv2.resize(aa, (int(aa.shape[1]/divide), int(aa.shape[0]/divide)))
print(f'shape: {aa.shape}')

# windows = []
# coords = []
# out = []
# for window, (x, y) in sliding_window_viola_jones(aa, 24):
#     window = np.array([window])
#     y_pred = c.predict(window)
#     if(y_pred[0] == 1):
#         out.append((x, y, window[0].shape[0]))


out = process(aa)


aa = cv2.cvtColor(aa, cv2.COLOR_GRAY2BGR)
aa = draw_squares(aa, out)
cv2.namedWindow('a', cv2.WINDOW_NORMAL)
cv2.resizeWindow('a', aa.shape[1], aa.shape[0])
cv2.imshow('a', aa)
cv2.imwrite('100.jpg', aa)
cv2.waitKey(0)
cv2.destroyAllWindows()