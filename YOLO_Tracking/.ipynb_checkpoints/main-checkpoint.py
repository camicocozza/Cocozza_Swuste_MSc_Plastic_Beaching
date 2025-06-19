import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import glob
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
# %matplotlib inline


from Settings import MODEL_PATH, VIDEO_ROOT, RESULTS_DIR, video_params, THRESHOLDS
from processing_trajectories import process_tracking_data
from Merging_Functions import merge_trajectories_by_time
from Merging_Functions import plot_all_merged_trajectories
from processing_trajectories import plot_all_trajectory_pieces
# from Creating_Final_Dict import build_final_dict
from Creating_Final_Dict import save_final_dict

# --- MAIN LOOP ---
all_trajectories = {}
model = YOLO(MODEL_PATH)
video_files = glob.glob(os.path.join(VIDEO_ROOT, "**", "*.mp4"), recursive=True)
video_files = [os.path.normpath(path) for path in video_files]

for video_path in tqdm(video_files, desc="Tracking videos"):
    file_name = os.path.basename(video_path)
    custom_name = os.path.splitext(file_name)[0]

    if file_name not in video_params:
        print(f"⚠️ Skipping {file_name} — missing parameters.")
        continue

    print(f"\n🔍 Tracking {custom_name}")
    print(f"   ▶ Source video: {video_path}")

    label_dir = os.path.join(RESULTS_DIR, custom_name, "labels")
    print(f"   🔎 Looking for label .txt files in: {label_dir}")

    params = video_params[file_name]
    
    if not os.path.exists(label_dir):
        try:
            for res in model.track(
                source=video_path,
                tracker="bytetrack.yaml",
                save=True,
                save_txt=True,
                save_conf=True,
                show=False,
                conf=0.3,
                stream=True,
                name=custom_name,
                project=RESULTS_DIR,
            ):
                result = res 
        except Exception as e:
            print(f"   ❌ Failed to process {custom_name}: {e}")
            result = None
        
        # if result is not None: 
        #     processed_shape = result.orig_img.shape  # (height, width, channels)
        #     height_resized = processed_shape[0]
        #     width_resized = processed_shape[1]
        #     print(f"Processed shape: height={height_resized}, width={width_resized}")
        #     ratio = height_resized/width_resized
        #     params = video_params[file_name].copy()
        #     params["ratio"] = ratio
    else:
        params = video_params[file_name]
        print(f'else {params}')
             

    if not os.path.exists(label_dir):
        print(f"   ❌ ERROR: Label directory does not exist.")
        continue

    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"   📂 Found {len(txt_files)} label files.")
    
    
    all_trajectories[custom_name] = process_tracking_data(label_dir, **params)
    print(f"   ✅ Processed {len(all_trajectories[custom_name])} particle tracks.\n")

# Save results
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, "all_trajectories.pkl")
with open(out_path, "wb") as f:
    pickle.dump(all_trajectories, f)

print(f"\nDone! Saved results to {out_path}")





###########################################################################################################################
# plotting the tracked trajectory pieces that result from running YOLO and are stored in all_trajectories. 
plot_all_trajectory_pieces(out_path)  # stored in processing_trajectories


# merged_trajectories_dic = merge_trajectories_by_time(all_trajectories, time_threshold=2, y_threshold=15)
merged_trajectories_dic = merge_trajectories_by_time(all_trajectories, time_threshold=THRESHOLDS["time_threshold"], y_threshold=THRESHOLDS["y_threshold"], min_x_loc1=THRESHOLDS["min_x_loc1"], min_x_loc2=THRESHOLDS["min_x_loc2"], min_x_loc3=THRESHOLDS["min_x_loc3"])
# print(merged_trajectories_dic)


plot_all_merged_trajectories(merged_trajectories_dic)


##########################################################################################################################
# Assume `new_dict` is created from your merging function for a new video
final_dict_path = os.path.join(RESULTS_DIR, "final_dict.pkl")
save_final_dict(merged_trajectories_dic, final_dict_path)

