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


from Settings import THRESHOLDS

def assign_trajectory_ids(df, distance_threshold=THRESHOLDS["distance_between_points"], max_frame_gap=THRESHOLDS["max_frame_gap"]):
    """
    The dataframe contains the frame number, time stamp and location of each particle. Based on framenumber of
    the datapoints and the proximity to the next datapoint the datapoints are linked together into trajecotry segments.
    
    input:
    - df: DataFrame with columns among others the ['Frame', 'Real_X', 'Real_Y']
    - distance_threshold: maximum distance (in cm) to consider the same trajectory (large enough the bridge small gaps where the 
    particle was lost by YOLO for a second, but small enough to distinguish between different particle trajecotries). 
    - max_frame_gap: max frame difference allowed between trajectory steps (large enough to bridge the small gaps, but small enough to
    make sure that they are still the same particle). 

    Returns:
    - DataFrame with added 'Track_ID' column
    """
    
    df = df.sort_values(by='Frame').reset_index(drop=True)
    df['traj_id'] = np.nan

    active_trajectories = [] 
    next_traj_id = 1

    # For each row it temporarely stores the required values
    for idx, row in df.iterrows():
        current_frame = row['Frame']
        current_x = row['Real_X']
        current_y = row['Real_Y']

        matched = False
        best_match_idx = None
        best_distance = float('inf')

        # Try to link particles together
        for i, (last_frame, last_x, last_y, traj_id) in enumerate(active_trajectories):
            frame_gap = current_frame - last_frame
            if frame_gap <= max_frame_gap: #####  0 <
                dist = np.sqrt((current_x - last_x)**2 + (current_y - last_y)**2)
                if dist < distance_threshold and dist < best_distance:
                    matched = True
                    best_distance = dist
                    best_match_idx = i

        if matched:
            # Assign existing traj_id
            _, _, _, matched_traj_id = active_trajectories[best_match_idx]
            df.at[idx, 'traj_id'] = matched_traj_id
            # Update trajectory
            active_trajectories[best_match_idx] = (current_frame, current_x, current_y, matched_traj_id)
        else:
            # Start a new trajectory
            df.at[idx, 'traj_id'] = next_traj_id
            active_trajectories.append((current_frame, current_x, current_y, next_traj_id))
            next_traj_id += 1

        # Optional: remove stale trajectories to save memory
        active_trajectories = [
            (f, x, y, tid) for (f, x, y, tid) in active_trajectories
            if current_frame - f <= max_frame_gap
        ]

    df['Track_ID'] = df['traj_id'].astype(int)
    return df






def filter_short_tracks(df, min_length=THRESHOLDS["min_frames"]):
    """ This filters out very short trajectory segments. This is required to get rid of tracking noise """
    filtered = []
    for track_id, group in df.groupby('Track_ID'):
        if len(group) >= min_length:
            filtered.append(group)
    if filtered:
        return pd.concat(filtered, ignore_index=True)
    else:
        return pd.DataFrame()




def resolve_duplicates(df):
    """
    Fixes duplicate Real_Time(s) values **within each Track_ID segment**.
    
    Sometimes a single particle is detected multiple times in one frame under different Class_IDs. 
    One particle is recognized as both yellow and green for example. This functions filters out the noise by checking 
    class id and dropping the incorrectly tracked buts.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with columns 'Track_ID', 'Real_Time(s)', and 'Class_ID'.

    Returns:
    pd.DataFrame: Cleaned DataFrame with within-Track_ID timestamp duplicates removed.
    """
    """Sometimes Tracker tracks ONE particle in ONE frame TWICE. Often, the colour of the particle is than incorrectly recognized. 
    One particle is recognized as both yellow and green for example. 
    
    This does not happen often, however duplicate timestemp values within one trajectory is problematic. 
     

    This function takes as input the dataframe, and returns the same dataframe without having Time stamp duplicates.
    """
    df = df.reset_index(drop=True).copy()
    to_drop = []

    for track_id, group in df.groupby('Track_ID'):
        group = group.reset_index()  # Keep original index for deletion
        duplicated_times = group['Real_Time(s)'][group['Real_Time(s)'].duplicated(keep=False)].unique()

        for t in duplicated_times:
            dup_rows = group[group['Real_Time(s)'] == t]
            dup_indices = dup_rows['index'].tolist()  # original df index

            # Skip if there's only one (as a check)
            if len(dup_indices) <= 1:
                continue

            # Get the previous index (within the group) to compare Class_ID
            first_local_idx = dup_rows.index.min()
            if first_local_idx == 0:
                # No previous row in the segment, keep first and drop others
                to_drop.extend(dup_indices[1:])
                continue

            prev_class = group.loc[first_local_idx - 1, 'Class_ID']

            # Try to keep the duplicate with the same Class_ID as the previous row
            kept = None
            '''
            for idx in dup_rows.index:
                if group.loc[idx, 'Class_ID'] == prev_class:
                    kept = group.loc[idx, 'index']
                    break'''

            # If no match found, just keep the first
            if kept is None:
                kept = dup_indices[0]

            # Drop all others
            for idx in dup_indices:
                if idx != kept:
                    to_drop.append(idx)

    df_cleaned = df.drop(index=to_drop).reset_index(drop=True)
    return df_cleaned





def clip_stabilized_segments(df, 
                             stabilization_threshold=THRESHOLDS["stabilization_threshold"], 
                             min_stabilization_frames=THRESHOLDS["min_stabilization_frames"], 
                             window_size=THRESHOLDS["window_size"]):
    """
    Removes the tail end of each trajectory segment once the particle has stabilized on the beach.
    Stabilization is determined using a moving-averaged positions to check over a certain amount of frames.

    Parameters:
    - df: DataFrame containing all trajectory segments
    - stabilization_threshold: maximum average distance in cm to consider the particle stabilized
    - min_stabilization_frames: number of consecutive stable frames needed to define stabilization
    - window_size: number of points to include in the moving average

    Returns:
    - A dataframe where each trajectory segment ends once the particle becomes stable.
    """
    
    clipped_segments = []

    for tid, group in df.groupby('Track_ID'):
        group = group.sort_values('Frame').reset_index(drop=True).copy()

        # Compute moving averages
        group['X_smooth'] = group['Real_X'].rolling(window=window_size, center=True, min_periods=1).mean()
        group['Y_smooth'] = group['Real_Y'].rolling(window=window_size, center=True, min_periods=1).mean()

        # Compute rolling distances
        dx = group['X_smooth'].diff()
        dy = group['Y_smooth'].diff()
        group['distance'] = np.sqrt(dx**2 + dy**2)

        # Identify stabilization point
        group['rolling_stable'] = group['distance'].rolling(window=min_stabilization_frames).mean()
        stabilized_idx = group[group['rolling_stable'] < stabilization_threshold].index

        if len(stabilized_idx) > 0:
            first_stable_idx = stabilized_idx[0]
            clipped = group.loc[:first_stable_idx - 1]  # exclude the first stable point and everything after
        else:
            clipped = group

        clipped_segments.append(clipped[['Real_Time(s)', 'Class_ID', 'Real_X', 'Real_Y', 'Track_ID']])

    if clipped_segments:
        return pd.concat(clipped_segments, ignore_index=True)
    else:
        return pd.DataFrame



def process_tracking_data(txt_dir, fps, real_width, ratio):
    """ The initial output of YOLO is a folder txt files. Each txt file represents a one frame of the  original video. Each file 
    contains the x and y location(s), class id and id-number of the particles dectect in the frame.  
    
    This functions opens the all txt files and stores the content of all files in one dataframe.
    
    The dataframe containing all datapoints tracked in one video (many particles per video) is then manipulated using the functions 
    above, to distinghish between different trajectories segments stored within the dataframe. 

    the input: 
    - txt_dir = directory where the txt files are stored. 
    - fps = video recordings frame rate (frames per second).
    - real_width = The real-world distance between the bottom left to bottom right corner of each video.
    - ratio = width to length ratio of each video 
    
    This function returns a dataframe containing trajectory segments representing each particle tracked within the video. 
    
    """

    data = []
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            frame_num = int(filename.split('_')[-1].split('.')[0])
            with open(os.path.join(txt_dir, filename), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        class_id = int(parts[0])
                        x_center, y_center = float(parts[1]), float(parts[2])
                        width, height = float(parts[3]), float(parts[4])
                        try:
                            id_number = float(parts[6])
                        except IndexError:
                            id_number = np.nan
                        real_time = frame_num / fps
                        data.append({
                            "Frame": frame_num,
                            "Real_Time(s)": real_time,
                            "Class_ID": class_id,
                            "X_Center": x_center,
                            "Y_Center": y_center,
                            "Width": width,
                            "Height": height,
                            "Id_number": id_number
                        })
    df = pd.DataFrame(data)
    print(f'dflength {len(df)}')
    df['Real_X'] = (np.abs(df['X_Center'] * real_width - real_width)) / 100
    df['Real_Y'] = ((real_width * ratio) - df['Y_Center'] * (real_width * ratio)) /100

    ######### This plots the original data for each video #########
    plt.figure(figsize=(14,8))
    plt.plot(df['Real_Time(s)'], df['Real_X'], '.')
    plt.title(f'Original Data {txt_dir}')


    ######### Manipulating the dataframe using the functions above #########
    df_clipped_framenr = assign_trajectory_ids(df)
    filtered = filter_short_tracks(df_clipped_framenr)
    without_timedubs = resolve_duplicates(filtered)
    # stabalised = clip_stabilized_segments(without_timedubs)
    final = without_timedubs[['Real_Time(s)', 'Class_ID', 'Real_X', 'Real_Y', 'Track_ID']]


    ######## Makes a dictionary per particle per segment #########
    particle_dict = {}
    for pid, group in final.groupby('Track_ID'):
        df_part = group[['Real_Time(s)', 'Real_X', 'Real_Y']].copy()
        df_part.columns = ['t', 'x', 'y']
        df_part = df_part.reset_index(drop=True)
        particle_dict[f'P{int(pid)}'] = df_part

    return final



############################################################################## PLOTTING INTERMEDIATE RESULTS

def plot_all_trajectory_pieces(pickle_path):
    """
    Loads trajectory data from a pickle file and plots all particle trajectories.
    
    """
    # Load the data
    plt.figure(figsize=(18, 6))
    with open(pickle_path, "rb") as f:
        all_trajectories = pickle.load(f)

    # Plot each trajectory
    for video_name, df in all_trajectories.items():
        print(f"🎬 Video: {video_name} — {df['Track_ID'].nunique()} trajectories")
        
        # Group by Track_ID (i.e., particle)
        for track_id, particle_df in df.groupby('Track_ID'):
            print(f"  📌 Track {track_id} with {len(particle_df)} points")
            
            # Plot the trajectory
            plt.plot(
                particle_df['Real_Time(s)'],
                particle_df['Real_X'],
                marker='.',
                alpha=0.7
            )

    plt.xlabel("Time (s)")
    plt.ylabel("X Position (cm)")
    plt.title("All Particle Trajectories after Processing")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize='small')
    plt.show()
