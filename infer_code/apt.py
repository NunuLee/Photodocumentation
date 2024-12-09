import cv2 
import os 
from model_build import build
from APT_DataLoader import load_data
from bestshot_selection import best_shot_selection, motion_detection
from model_inference import apt_inference

CLASS_NAME = ['01_Esoph','02_Esoph_junc','03_Duo_2nd_por','04_Duod_bulb','05_Antrum','06_Angle','07_Body_LB-MB','08_Body_MB-HB','09_Retro-Fundus','10_Retro-Cardia','11_Retro-LCside', '12_NA']


if __name__ == "__main__":
    apt_model_path = 'TRT model path'
    
    apt_model = build(apt_model_path)
             
    img_dir = 'input image directory path'
    des_dir = 'output root directory path'
    
    pre_frame = None 
    landmark_buffer = [[] for i in range(12)]
    
    files = os.listdir(img_dir)
    files.sort()
    
    for file_cnt, file in enumerate(files):
        print(f'{file_cnt:05d} / {len(files):05d}', end = '\r')
        
        file_name, ext = os.path.splitext(file)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            print(f"{file} has the wrong extension. Please check your image file again.")
            continue

        if not (os.path.exists(des_dir)) :
            for i in range(12):
                os.makedirs(os.path.join(des_dir, CLASS_NAME[i]))
       
        frame = cv2.imread(os.path.join(img_dir, file))
        md = motion_detection(pre_frame, frame)
        pre_frame = frame.copy() 
        
        dataloader224 = load_data(frame, 224)
        
        apt_class = apt_inference(dataloader224, apt_model)

        if apt_class >= 11 : # class 12 - 18 frames are grouped into a single 'NA' class
            landmark_buffer[11].append([frame, file_cnt, md])
        else:
            landmark_buffer[apt_class].append([frame, file_cnt, md])

    print("Best Shot Selection for each Landmark")
    for landmark, landmark_frames in enumerate(landmark_buffer[:11]) : 
        best_shot_result = best_shot_selection(landmark_frames)
        
        for i, result in enumerate(best_shot_result):
            frame = result[0]
            frame_cnt = result[1]
            cv2.imwrite(os.path.join(des_dir, CLASS_NAME[landmark], f'landmark{landmark+1}_{i}_frame{frame_cnt:05d}.jpg'), frame)

    print('APT completed')