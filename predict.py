
import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   If you want to modify the color of the corresponding type, you can modify self. colors in the __init__ function
    #-------------------------------------------------------------------------#
    deeplab = DeeplabV3()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           Indicates a single image prediction. If you want to modify the prediction process, such as saving the image, cropping objects, etc., you can first refer to the detailed comments below
    #   'video'             Indicates video detection, which can be performed by calling the camera or video. For details, please refer to the comments below.
    #   'fps'               Indicates test fps, the image used is street.jpg from img. Please refer to the comments below for details.
    #   'dir_predict'       Traverse the folder for detection and saving. By default, traverse the img folder and save the imd_out folder. For details, please refer to the comments below.
    #   'export_onnx'       To export the model as onnx, pytorch 1.7.1 or higher is required.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   count               Designated whether to perform pixel counting (i.e. area) and proportional calculation for the target
    #   name_classes        The types of differentiation are the same as those in json-toudataset, used for printing types and quantities
    #
    #   count、name_classes are only valid when mode='predict '
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Used to specify the path of the video, when video_cath=0, it indicates the detection of the camera
    #                       If you want to detect videos, you can set it to video_cath="xxx. mp4", which means reading the xxx.mp4 file from the root directory.
    #   video_save_path     Indicates the path for saving the video. When video_stave_math="", it means not to save
    #                       If you want to save the video, you can set it to 'yyy. mp4', which means saving it as a yyy.mp4 file in the root directory.
    #   video_fps           Fps for saved videos
    #
    #   video_path、video_save_path and video_fps are only valid when mode='video'
    #   When saving a video, you need to use Ctrl+C to exit or run until the last frame to complete the complete save process.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Used to specify the number of image detections when measuring fps. In theory, the larger the test_interval, the more accurate the fps.
    #   fps_image_path      Used to specify the fps image for testing
    #   
    #   test_interval and fps_image_math are only valid when mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     The folder path for the images used for detection has been specified
    #   dir_save_path       Designated the save path for the detected image
    #   
    #   dir_originpath and dir_stave_path are only valid when mode='dir_predict '
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify            Use Simplify onnx
    #   onnx_save_path      Designated the save path for onnx
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            # img = r"C:\0image_train\JPEGImages\a21_31_3_9.png"
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to correctly read the camera (video), please pay attention to whether the camera is installed correctly (whether the video path is filled in correctly)")

        fps = 0.0
        while(True):
            t1 = time.time()

            ref, frame = capture.read()
            if not ref:
                break
            # BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Image
            frame = Image.fromarray(np.uint8(frame))

            frame = np.array(deeplab.detect_image(frame))

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
