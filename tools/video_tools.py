import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image
import numpy as np     
from moviepy.editor import *
import os
 
 
def Pic2Video():
    imgPath = "/data/yuanhaolei/prox_children_S12/recordings/MoCap_20230422_150412/Color/"  # 读取图片路径
    videoPath = "output/S12/S12_color.mp4"  # 保存视频路径
 
    images = os.listdir(imgPath)
    fps = 15  # 每秒15帧数
 
    # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
 
    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    
    # for im_name in range(0, len(images)):
    for im_name in range(49, 135):
        frame = cv2.imread(imgPath + f'{im_name:03d}.png')  # 这里的路径只能是英文路径
        
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
        print(im_name)
        videoWriter.write(frame)
    print("图片转视频结束！")
    videoWriter.release()
    cv2.destroyAllWindows()

# 读取要合并的视频文件
def video_concat():
    sub_ids = 'S01'
    videoLeftUp = cv2.VideoCapture(f'/home/yuanhaolei/Document/code/pressure_toolkit/output/{sub_ids}/{sub_ids}_color.mp4')
    videoLeftDown = cv2.VideoCapture(f'/home/yuanhaolei/Document/code/pressure_toolkit/output/{sub_ids}/ours_blender.mkv')
    videoRightUp = cv2.VideoCapture(f'/home/yuanhaolei/Document/code/pressure_toolkit/output/{sub_ids}/prox_blender.mkv')
    videoRightDown = cv2.VideoCapture(f'/home/yuanhaolei/Document/code/pressure_toolkit/output/{sub_ids}/lemo_blender.mkv')

    fps = videoLeftUp.get(cv2.CAP_PROP_FPS)

    # 帧宽度和帧高度设置原视频的一倍
    width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH))) * 2
    height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT))) * 2
    # 视频合并完成存放的位置
    videoWriter = cv2.VideoWriter(f'/home/yuanhaolei/Document/code/pressure_toolkit/output/{sub_ids}/concat_result_{sub_ids}_blender.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

    successLeftUp, frameLeftUp = videoLeftUp.read()
    successLeftDown, frameLeftDown = videoLeftDown.read()
    successRightUp, frameRightUp = videoRightUp.read()
    successRightDown, frameRightDown = videoRightDown.read()

    while successLeftUp and successLeftDown and successRightUp and successRightDown:
        frameLeftUp = cv2.resize(frameLeftUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(frameLeftDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameRightUp = cv2.resize(frameRightUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameRightDown = cv2.resize(frameRightDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

        frameUp = np.hstack((frameLeftUp, frameRightUp))
        frameDown = np.hstack((frameLeftDown, frameRightDown))
        frame = np.vstack((frameUp, frameDown))

        videoWriter.write(frame)
        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown, frameLeftDown = videoLeftDown.read()
        successRightUp, frameRightUp = videoRightUp.read()
        successRightDown, frameRightDown = videoRightDown.read()

    videoWriter.release()
    videoLeftUp.release()
    videoLeftDown.release()
    videoRightUp.release()
    videoRightDown.release()


def concat_video_temp():    
    # 定义一个数组
    video_path_list = ['/home/yuanhaolei/Document/code/pressure_toolkit/output/S01/concat_result_S01.mp4',
         '/home/yuanhaolei/Document/code/pressure_toolkit/output/S07/concat_result_S07.mp4',
         '/home/yuanhaolei/Document/code/pressure_toolkit/output/S10/concat_result_S10.mp4',
         '/home/yuanhaolei/Document/code/pressure_toolkit/output/S11/concat_result_S11.mp4',
         '/home/yuanhaolei/Document/code/pressure_toolkit/output/S12/concat_result_S12.mp4']
    video_list = []
    for i in range(len(video_path_list)):
        video = VideoFileClip(video_path_list[i])
        video_list.append(video)
    # 拼接视频
    final_clip = concatenate_videoclips(video_list)

    # 生成目标视频文件
    final_clip.to_videofile("/home/yuanhaolei/Document/code/pressure_toolkit/output/output_concat_temp.mp4", fps=15, remove_temp=False)


if __name__ == "__main__":
    # Pic2Video()
    video_concat()
    # concat_video_temp()