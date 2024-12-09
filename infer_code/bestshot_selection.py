import cv2 
import numpy as np 

""" 
landmark_buffer = [
                        [frame(array), frame_cnt(int), md(bool)],
                        [frame(array), frame_cnt(int), md(bool)],
                        [frame(array), frame_cnt(int), md(bool)],
                        ...
                    ]

"""
def motion_detection(frame_pre, frame):
    
    if (frame_pre is not None):
        
        frame_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)
        frame_pre = cv2.resize(frame_pre, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pixels = frame.shape[0] * frame.shape[1]
        
        diff1 = cv2.absdiff(frame_pre, frame)
        ret, diff1_t = cv2.threshold(diff1, 15, 255, cv2.THRESH_BINARY)
        diff_cnt1 = cv2.countNonZero(diff1_t)
        
        return diff_cnt1 / pixels 
    else:
        return -1

def detect_blur_fft(image, size=40):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)     
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    
    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
   
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean

def best_shot_selection(landmark_buffer) :
    result_list = []

    for i in range(len(landmark_buffer)):
        frame = landmark_buffer[i][0]
        frame_cnt = landmark_buffer[i][1]
        md = landmark_buffer[i][2]
        fft = detect_blur_fft(frame)
        result_list.append([frame, frame_cnt, md, fft])
        
    sorted_result_list = sorted(
        result_list,
        key = lambda x : (
            x[2],               # Sort by motion_detection in ascending order
            -x[3]               # Sort by fft in descending order
        )
    )
    
    filtered_result_list = []
    frame_cnts_included = set()
    
    for result in sorted_result_list : 
        frame_cnt = result[1]
        # check if this frame_cnt has no other within a difference of 50
        if all(abs(frame_cnt - included) >= 50 for included in frame_cnts_included):
            filtered_result_list.append(result)
            frame_cnts_included.add(frame_cnt)
            
    return filtered_result_list
