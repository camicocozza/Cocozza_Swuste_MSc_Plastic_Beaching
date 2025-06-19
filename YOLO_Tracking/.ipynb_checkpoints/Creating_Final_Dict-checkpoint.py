# import os
# import pickle

# def build_final_dict(merged_trajectories, save_path):
#     """
#     Builds or updates the final dictionary using a file on disk.

#     Parameters:
#     - merged_trajectories (dict): The result of merge_all_trajectories.
#     - save_path (str): Path to the pickle file storing the final_dict.

#     Returns:
#     - dict: Updated final_dict.
#     """
#     # Load existing final_dict from file if it exists
#     if os.path.exists(save_path):
#         with open(save_path, "rb") as f:
#             final_dict = pickle.load(f)
#         print(f"📂 Loaded existing final_dict from: {save_path}")
#     else:
#         final_dict = {}
#         print(f"📄 No existing file. Creating new final_dict.")

#     # Update logic
#     for key, trajectories in merged_trajectories.items():
#         try:
#             wave, run, weightset = key.split('_')
#         except ValueError:
#             print(f"⚠️ Unexpected key format: {key}. Skipping.")
#             continue

#         final_key = f"{wave}_{run}_{weightset}"

#         if final_key not in final_dict:
#             final_dict[final_key] = {
#                 "wave_height": wave,
#                 "run": run,
#                 "weightset": weightset,
#                 "merged_trajectories": {}
#             }

#         for class_id, segments in trajectories.items():
#             for i, seg in enumerate(segments):
#                 unique_key = f"Class_{class_id}_{i}"

#                 if unique_key not in final_dict[final_key]["merged_trajectories"]:
#                     final_dict[final_key]["merged_trajectories"][unique_key] = seg
#                 else:
#                     print(f"🔁 Skipping duplicate segment: {final_key} → {unique_key}")

#     # Save updated final_dict back to file
#     with open(save_path, "wb") as f:
#         pickle.dump(final_dict, f)
#     print(f"💾 Final_dict saved to: {save_path}")

#     return final_dict










# import os
# import pickle

# def save_final_dict(new_data: dict, save_path: str):
#     """
#     Save or update a dictionary to a pickle file without duplicating top-level keys.

#     Parameters:
#     - new_data (dict): The newly generated dictionary to add.
#     - save_path (str): Path to the pickle file where the dictionary should be saved.
#     """
#     # Load existing data if file exists
#     if os.path.exists(save_path):
#         with open(save_path, 'rb') as f:
#             final_dict = pickle.load(f)
#     else:
#         final_dict = {}

#     # Add only new top-level keys (e.g., Hx_Rx) that aren't already in the file
#     for top_key, value in new_data.items():
#         if top_key not in final_dict:
#             final_dict[top_key] = value
#         else:
#             print(f"Skipping duplicate entry for top key: {top_key}")

#     # Save back to the file
#     with open(save_path, 'wb') as f:
#         pickle.dump(final_dict, f)

#     print(f"Dictionary saved to {save_path}")





import os
import pickle

def save_final_dict(new_data, save_path):
    # Check if file exists and is non-empty
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with open(save_path, 'rb') as f:
            try:
                final_dict = pickle.load(f)
            except EOFError:
                print("⚠️ Warning: final_dict.pkl was empty or corrupted. Creating a new dictionary.")
                final_dict = {}
    else:
        final_dict = {}

    # Overwrite existing entries with new data
    for key, value in new_data.items():
        final_dict[key] = value  # overwrite regardless of existing content

    # Save the updated dictionary
    with open(save_path, 'wb') as f:
        pickle.dump(final_dict, f)


