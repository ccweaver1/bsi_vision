
import os
import cv2
from tqdm import tqdm

# video_name = 'DET-NSH_10_21_16.mp4'
video_name = 'BUF-EDM_10-16-16.mp4'
starting_point_tup = (1, 53, 26) # starting point in (hours, minutes, seconds)
h, m, s = starting_point_tup
capture_length = 20 # length of capture in seconds 
gap_between_frames = 5 # capture every x-th frame

starting_point = h*3600 + m*60 + s
ending_point = starting_point + capture_length
h_end = int(ending_point / 3600)
m_end = int((ending_point - h_end*3600) / 60)
s_end = int((ending_point - h_end*3600 - m_end*60))

base_path = '/Users/campbellweaver/Documents/VisionData/'
path_to_video = os.path.join(base_path, 'videos', video_name)
output_folder_name = "{}_{}h{}m{}s-{}h{}m{}s".format(video_name.split('_')[0], h,m,s,h_end,m_end,s_end)
output_save_path = os.path.join(base_path, output_folder_name, 'frames')
if not os.path.exists(output_save_path):
    os.mkdir(output_save_path)


video_capture = cv2.VideoCapture(path_to_video)
video_capture.set(cv2.CAP_PROP_POS_MSEC, float(starting_point*1000.0))
fps = video_capture.get(cv2.CAP_PROP_FPS)
starting_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
ending_frame = int(starting_frame + capture_length*fps)


#if we wanted to write it back to a video, we'd use a video_writer
# fps = video_capture.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_output = cv2.VideoWriter(output_save_path, fourcc, fps, (1280, 720))

for frame_num in tqdm(range(starting_frame, ending_frame, gap_between_frames)):   
    (grabbed, frame) = video_capture.read()

    if not grabbed:
        print('EOF (end of film)')
        break
    
    frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
    img_path = os.path.join(output_save_path, video_name.split('.')[0] + '_' + str(int(frame_number)) + '.png')
    cv2.imwrite(img_path, frame)
    for _ in range(gap_between_frames - 1):
        grabbed = video_capture.grab()

    # video_output.write(resized)

    