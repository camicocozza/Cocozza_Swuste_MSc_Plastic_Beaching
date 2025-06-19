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




# ######################## With X lenght Filter for each individual trajectory bit ###############################

# def merge_trajectories_by_time(all_trajectories, time_threshold, y_threshold, min_x_loc1, min_x_loc2, min_x_loc3):
#     import re
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Pattern matching
#     H_pattern = r'H(\d+)'
#     R_pattern = r'R(\d+)'
#     W_pattern = r'W([A-Z]+)'  # WL, WA, WH

#     grouped_by_HRW = {}
#     merged_trajectories_by_HRW = {}

#     for video_name, df in all_trajectories.items():
#         H_match = re.search(H_pattern, video_name)
#         R_match = re.search(R_pattern, video_name)
#         W_match = re.search(W_pattern, video_name)

#         if H_match and R_match and W_match:
#             H_value = H_match.group(1)
#             R_value = R_match.group(1)
#             W_value = W_match.group(1)

#             HRW_key = f"H{H_value}_R{R_value}_W{W_value}"

#             if HRW_key not in grouped_by_HRW:
#                 grouped_by_HRW[HRW_key] = []

#             grouped_by_HRW[HRW_key].append((video_name, df))
#         else:
#             print(f"⚠️ Could not extract H, R, or W from: {video_name}")

#     for HRW_key, video_dfs in grouped_by_HRW.items():
#         df_loc1 = df_loc2 = df_loc3 = None

#         for video_name, df in video_dfs:
#             if 'LOC1' in video_name:
#                 df_loc1 = df
#             elif 'LOC2' in video_name:
#                 df_loc2 = df
#             elif 'LOC3' in video_name:
#                 df_loc3 = df

#         if df_loc1 is None or df_loc2 is None or df_loc3 is None:
#             print(f"⚠️ Missing LOCs for {HRW_key}: LOC1: {df_loc1 is not None}, LOC2: {df_loc2 is not None}, LOC3: {df_loc3 is not None}")
#             continue

#         # Optional: Plot before merging
#         loc_dfs = {'LOC1': df_loc1, 'LOC2': df_loc2, 'LOC3': df_loc3}
#         for loc_label, loc_df in loc_dfs.items():
#             plt.figure(figsize=(10, 4))
#             plt.plot(loc_df['Real_Time(s)'], loc_df['Real_X'], '.')
#             plt.xlabel('Real_Time(s)')
#             plt.ylabel('Real_X')
#             plt.ylim(0, 350)
#             plt.yticks(np.arange(0, 351, 50))
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()

#         df_loc1_grouped = df_loc1.sort_values('Real_Time(s)').groupby('Track_ID')
#         df_loc2_grouped = df_loc2.sort_values('Real_Time(s)').groupby('Track_ID')
#         df_loc3_grouped = df_loc3.sort_values('Real_Time(s)').groupby('Track_ID')

#         merged_trajectories_by_HRW[HRW_key] = {}

#         for loc1_track_id, loc1_group in df_loc1_grouped:
#             x_length_loc1 = loc1_group['Real_X'].max() - loc1_group['Real_X'].min()
#             if x_length_loc1 < min_x_loc1:
#                 # print(f'x_length_loc1 {x_length_loc1}')
#                 continue  # Skip this trajectory

#             last_time_loc1 = loc1_group['Real_Time(s)'].iloc[-1]
#             last_y_loc1 = loc1_group['Real_Y'].iloc[-1]

#             for loc2_track_id, loc2_group in df_loc2_grouped:
#                 x_length_loc2 = loc2_group['Real_X'].max() - loc2_group['Real_X'].min()
#                 if x_length_loc2 < min_x_loc2:
#                     # print(f'x_length_loc2 {x_length_loc2}')
#                     continue

#                 first_time_loc2 = loc2_group['Real_Time(s)'].iloc[0]
#                 first_y_loc2 = loc2_group['Real_Y'].iloc[0]

#                 if abs(last_time_loc1 - first_time_loc2) <= time_threshold and abs(last_y_loc1 - first_y_loc2) <= y_threshold:
#                     last_x_loc1 = loc1_group['Real_X'].iloc[-1]
#                     loc2_group = loc2_group.copy()
#                     loc2_group['Real_X'] += last_x_loc1
#                     merged_group = pd.concat([loc1_group, loc2_group], ignore_index=True)

#                     for loc3_track_id, loc3_group in df_loc3_grouped:
#                         x_length_loc3 = loc3_group['Real_X'].max() - loc3_group['Real_X'].min()
#                         if x_length_loc3 < min_x_loc3:
#                             # print(f'x_length_loc3 {x_length_loc3}')
#                             continue

#                         first_time_loc3 = loc3_group['Real_Time(s)'].iloc[0]
#                         first_y_loc3 = loc3_group['Real_Y'].iloc[0]

#                         if abs(merged_group['Real_Time(s)'].iloc[-1] - first_time_loc3) <= time_threshold and abs(merged_group['Real_Y'].iloc[-1] - first_y_loc3) <= y_threshold:
#                             last_x_merged = merged_group['Real_X'].iloc[-1]
#                             loc3_group = loc3_group.copy()
#                             loc3_group['Real_X'] += last_x_merged

#                             final_merged_group = pd.concat([merged_group, loc3_group], ignore_index=True)

#                             class_id = loc1_group['Class_ID'].iloc[0]
#                             if class_id not in merged_trajectories_by_HRW[HRW_key]:
#                                 merged_trajectories_by_HRW[HRW_key][class_id] = []

#                             merged_trajectories_by_HRW[HRW_key][class_id].append(final_merged_group) ############## check 

#     return merged_trajectories_by_HRW



################################# CAMILLA VERSION ########################################################################

def merge_trajectories_by_time(all_trajectories, time_threshold, y_threshold, min_x_loc1, min_x_loc2, min_x_loc3):
    """ This function has as its main goal to merge the trajectory segments from the 2 or 3 consequtive videos to trajectories of individual particles for the entire length of the flume. 
    This is done by first matching the Run and Wave condition together to find the 2/3 topview videos that belong together. 
    Then dataframes made for each of these videos are grouped based on their Track_ID (individual segments). After which the segments are linked together based on certain checks. 

    The following things are checked to merge the right segments: 
    - The length of each segment in x direction, to ensure that it is a complete segment. 
    - The difference in time and the difference in y location between the start of the segment from video made at location 2 with the end of the segment from video at location 1 (and 3 and 2 respectively)
    
    input: 
    - all_trajectories: the dictionary with the datapoints. 
    - time_threshold: the maximum accepted jump in time between the videos (result of cropping the videos)
    - y_threshold:  the maximum accepted jump in y location between the videos (result of cropping the videos)
    - min_x_loc1 2 and 3: the minimum length of the travel distance (cm)

    This results in a dictionary that stores all individual particles trajectories. 
    """
    
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    H_pattern = r'H(\d+)'
    R_pattern = r'R(\d+)'
    W_pattern = r'W([A-Z]+)'  # WL, WA, WH

    grouped_by_HRW = {}
    merged_trajectories_by_HRW = {}

    for video_name, df in all_trajectories.items():
        H_match = re.search(H_pattern, video_name)
        R_match = re.search(R_pattern, video_name)
        W_match = re.search(W_pattern, video_name)

        if H_match and R_match and W_match:
            H_value = H_match.group(1)
            R_value = R_match.group(1)
            W_value = W_match.group(1)

            HRW_key = f"H{H_value}_R{R_value}_W{W_value}"

            if HRW_key not in grouped_by_HRW:
                grouped_by_HRW[HRW_key] = []

            grouped_by_HRW[HRW_key].append((video_name, df))
        else:
            print(f"⚠️ Could not extract H, R, or W from: {video_name}")

    for HRW_key, video_dfs in grouped_by_HRW.items():
        df_loc1 = df_loc2 = df_loc3 = None

        for video_name, df in video_dfs:
            if 'LOC1' in video_name:
                df_loc1 = df
            elif 'LOC2' in video_name:
                df_loc2 = df
            elif 'LOC3' in video_name:
                df_loc3 = df
                print("saving df_loc3")

        if df_loc1 is None or df_loc2 is None:
            print(f"⚠️ Missing LOCs for {HRW_key}: LOC1: {df_loc1 is not None}, LOC2: {df_loc2 is not None}")
        else:

      
            '''
            # Plot LOC1 and LOC2
            for loc_label, loc_df in {'LOC1': df_loc1, 'LOC2': df_loc2}.items():
                plt.figure(figsize=(10, 4))
                plt.plot(loc_df['Real_Time(s)'], loc_df['Real_X'])
                plt.xlabel('Real_Time(s)')
                plt.ylabel('Real_X')
                plt.ylim(0, 3.50)
                plt.yticks(np.arange(0, 351, 50))
                plt.grid(True)
                plt.tight_layout()
                plt.title(f'Video {video_name}')
                plt.show()'''
    
            df_loc1_grouped = df_loc1.sort_values('Real_Time(s)').groupby('Track_ID')
            df_loc2_grouped = df_loc2.sort_values('Real_Time(s)').groupby('Track_ID')
    
            '''
            # Plot LOC1 grouped trajectories
            plt.figure(figsize=(10, 4))
            for track_id, group in df_loc1_grouped:
                plt.plot(group['Real_Time(s)'], group['Real_X'], label=f'Track {track_id}')
            plt.xlabel('Real_Time(s)')
            plt.ylabel('Real_X')
            plt.title('LOC1 Trajectories')
            plt.ylim(0, 3.50)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Plot LOC2 grouped trajectories
            plt.figure(figsize=(10, 4))
            for track_id, group in df_loc2_grouped:
                plt.plot(group['Real_Time(s)'], group['Real_X'], label=f'Track {track_id}')
            plt.xlabel('Real_Time(s)')
            plt.ylabel('Real_X')
            plt.title('LOC2 Trajectories')
            plt.ylim(0, 3.50)
            plt.grid(True)
            plt.tight_layout()
            plt.show()'''
    
            print("LOC1, number of trajectories:", len(df_loc1_grouped))
            print("LOC2, number of trajectories:", len(df_loc2_grouped))
            merged_trajectories_by_HRW[HRW_key] = []
    
            
    
            for loc1_track_id, loc1_group in df_loc1_grouped:
                x_length_loc1 = loc1_group['Real_X'].max() - loc1_group['Real_X'].min()
                
                if x_length_loc1 < min_x_loc1:
                    #print(f"Failure due to x_length loc 1 too short {x_length_loc1}, threshold: {min_x_loc1}")
                    continue
                dt = loc1_group['Real_Time(s)'].iloc[4] - loc1_group['Real_Time(s)'].iloc[3]
                
                last_time_loc1 = loc1_group['Real_Time(s)'].iloc[-1]
                last_y_loc1 = loc1_group['Real_Y'].iloc[-1]
    
                for loc2_track_id, loc2_group in df_loc2_grouped:
                    x_length_loc2 = loc2_group['Real_X'].max() - loc2_group['Real_X'].min()
                    if x_length_loc2 < min_x_loc2:
                        #print(f"Failure due to x_length loc 2 too short {x_length_loc2}, threshold: {min_x_loc2}")
                        continue
    
                    first_time_loc2 = loc2_group['Real_Time(s)'].iloc[0]
                    first_y_loc2 = loc2_group['Real_Y'].iloc[0]
    
                    if abs(last_time_loc1 - first_time_loc2) <= time_threshold and abs(last_y_loc1 - first_y_loc2) <= y_threshold:
                        last_x_loc1 = loc1_group['Real_X'].iloc[-1]
    
                        loc2_group = loc2_group.copy()
                        loc2_group['Real_X'] += last_x_loc1 
                        time_gap = first_time_loc2 - last_time_loc1 #added
                        
                        loc2_group['Real_Time(s)'] =  loc2_group['Real_Time(s)'] - time_gap + dt
                        merged_group = pd.concat([loc1_group, loc2_group], ignore_index=True)
    
                        merged_trajectories_by_HRW[HRW_key].append(merged_group)
                        #else:
                        #print(f"Failure due to time: {abs(last_time_loc1 - first_time_loc2)}, threshold: {time_threshold}")
                        #print(f"Failure due to y: {abs(last_y_loc1 - first_y_loc2)}, threshold: {y_threshold}")

        # Add LOC3 separately under a new key
        if df_loc3 is not None:
            loc3_key = HRW_key + '_LOC3'
            merged_trajectories_by_HRW[loc3_key] = []
            df_loc3_grouped = df_loc3.sort_values('Real_Time(s)').groupby('Track_ID')
            for _, loc3_group in df_loc3_grouped:
                x_length_loc3 = loc3_group['Real_X'].max() - loc3_group['Real_X'].min()
                if x_length_loc3 >= min_x_loc3:
                    
                    merged_trajectories_by_HRW[loc3_key].append(loc3_group)

    return merged_trajectories_by_HRW



def plot_all_merged_trajectories(merged_trajectories_by_HRW):
    """
    Plots all merged trajectories for each Hx_Rx_WX key.
    """
    for hrw_key, trajectory_list in merged_trajectories_by_HRW.items():
        if not isinstance(trajectory_list, list):
            continue  # Skip unexpected entries

        # --- Plot 1: Real_X over Time ---
        plt.figure(figsize=(12, 5))
        plt.title(f"Real_X vs Real_Time(s) for {hrw_key}")
        plt.xlabel("Real_Time(s)")
        plt.ylabel("Real_X")

        for i, df in enumerate(trajectory_list):
            plt.plot(df["Real_Time(s)"], df["Real_X"], label=f"Segment {i}")

        plt.grid(True)
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Real_Y over Time ---
        plt.figure(figsize=(12, 5))
        plt.title(f"Real_Y vs Real_Time(s) for {hrw_key}")
        plt.xlabel("Real_Time(s)")
        plt.ylabel("Real_Y")

        for i, df in enumerate(trajectory_list):
            plt.plot(df["Real_Time(s)"], df["Real_Y"], label=f"Segment {i}")

        plt.grid(True)
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.show()

        # --- Plot 3: Real_Y vs Real_X (Top-Down View) ---
        plt.figure(figsize=(10, 6))
        plt.title(f"Real_Y vs Real_X (Top-Down) for {hrw_key}")
        plt.xlabel("Real_X")
        plt.ylabel("Real_Y")

        for i, df in enumerate(trajectory_list):
            plt.plot(df["Real_X"], df["Real_Y"], label=f"Segment {i}")

        plt.grid(True)
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

