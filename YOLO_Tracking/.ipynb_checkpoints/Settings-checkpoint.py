# --- USER PATH CONFIGURATION ---
MODEL_PATH = "/Volumes/CAMI_2/YOLO_trajectories/runs/detect/train_last/weights/best.pt" # path to YOLO model
VIDEO_ROOT = "/Volumes/CAMI_2/VIDEOS" # Path to folder in which the videos are stored
RESULTS_DIR = "/Volumes/CAMI_2/RESULTS" # path to folder where the results should be saved

# --- VIDEO PARAMETERS ---
""" For each video in the VIDEO_ROOT folder, the parameters should be stored. 
This includes 
- fps = number of frams per second
- real_width = Real-World distance of view of camera left to right in centimetres
- ratio = ratio between Height to width of videos (so 164 pixels high against 640 pixels wide)

If for any video stored in VIDEO_ROOT the parameters are not found, the code stops running yielding 
no results"""

video_params = {

    "H5_LOC1_R1_WA.mp4": {"fps": 24, "real_width": 299, "ratio": 164/640}, 
    "H5_LOC2_R1_WA.mp4": {"fps": 24, "real_width": 191, "ratio": 260/640},

    "H5_LOC1_R2_WA.mp4": {"fps": 24, "real_width": 299, "ratio": 162/640},
    "H5_LOC2_R2_WA.mp4": {"fps": 24, "real_width": 191, "ratio": 265/640},

    "H5_LOC1_R3_WA.mp4": {"fps": 24, "real_width": 299, "ratio": 160/640}, 
    "H5_LOC2_R3_WA.mp4": {"fps": 24, "real_width": 191, "ratio": 263/640},

    "H5_LOC1_R5_WA.mp4": {"fps": 24, "real_width": 299, "ratio": 163/640}, 
    "H5_LOC2_R5_WA.mp4": {"fps": 24, "real_width": 160, "ratio": 315/640},

    "H5_LOC1_R6_WA.mp4": {"fps": 24, "real_width": 299, "ratio": 163/640}, 
    "H5_LOC2_R6_WA.mp4": {"fps": 24, "real_width": 160, "ratio": 322/640},
    
    "H8_LOC1_R1_WA.mp4": {"fps": 24, "real_width": 305, "ratio": 160/640},
    "H8_LOC2_R1_WA.mp4": {"fps": 24, "real_width": 290, "ratio": 171/640},
    
    "H8_LOC1_R2_WA.mp4": {"fps": 24, "real_width": 305, "ratio": 159/640},
    "H8_LOC2_R2_WA.mp4": {"fps": 24, "real_width": 290, "ratio": 172/640},
    
    "H8_LOC1_R3_WA.mp4": {"fps": 24, "real_width": 305, "ratio": 160/640}, 
    "H8_LOC2_R3_WA.mp4": {"fps": 24, "real_width": 290, "ratio": 170/640},

    "H8_LOC1_R4_WA.mp4": {"fps": 24, "real_width": 305, "ratio": 192/640}, 
    "H8_LOC2_R4_WA.mp4": {"fps": 24, "real_width": 290, "ratio": 192/640},
    
    "H11_LOC1_R8_WA.mp4": {"fps": 30, "real_width": 287, "ratio": 192/640}, 
    "H11_LOC2_R8_WA.mp4": {"fps": 30, "real_width": 222, "ratio": 224/640}, 
    "H11_LOC3_R8_WA.mp4": {"fps": 30, "real_width": 268, "ratio": 192/640}, 

    "H11_LOC1_R9_WA.mp4": {"fps": 30, "real_width": 287, "ratio": 192/640},
    "H11_LOC2_R9_WA.mp4": {"fps": 30, "real_width": 219, "ratio": 224/640}, 
    "H11_LOC3_R9_WA.mp4": {"fps": 30, "real_width": 267, "ratio": 192/640} 
    
    "H11_LOC1_R10_WA.mp4": {"fps": 30, "real_width": 287, "ratio": 192/640},
    "H11_LOC2_R10_WA.mp4": {"fps": 30, "real_width": 220, "ratio": 224/640},
    "H11_LOC3_R10_WA.mp4": {"fps": 30, "real_width": 268, "ratio": 192/640}
    
}

""" During the processing of the initial YOLO Data, and the merging of video segments, threshold values are used based on which 
data is kept or rejected"""

# --- THRESHOLDS FOR PROCESSING AND MERGING THE DATA---
THRESHOLDS = {
    # PROCESSING: 
    "distance_between_points": 0.5, # [cm] max distance (x and y considered) between two datapoints that can still be considered to be the same trajectory. 
    "max_frame_gap": 100, # max number of frames between two datapoints that can still be considered to be the same trajectory. (300 frames is +- 5 sec)
    "min_frames": 40, # minimum number of datapoints to be considered a trajectory segment. 

    
    "stabilization_threshold": 0.1, # [cm] Distance in cm between moving averaged points to consider "stable."
    "min_stabilization_frames": 50, #  how long (in frames) the particle must remain stable before clipping the rest
    "window_size": 20, # size of the rolling average window over which it checks weather it is stable

    # MERGING: 
    "time_threshold": 10, # the acceptable jump in time in the merging of two videos (due to trimming videos)
    "y_threshold": 0.15, # the acceptable jump in y direction in the merging of two videos (due to trimming videos)
    "min_x_loc1": 1, # the minimal length in x direction a segment has to be to 
    "min_x_loc2": 1.5, 
    "min_x_loc3": 1
}
