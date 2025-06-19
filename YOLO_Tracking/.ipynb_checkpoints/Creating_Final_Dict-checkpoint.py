"""
Utility function to update and save the final dictionary to a pickle file.

- If the file exists and is non-empty, loads the existing dictionary.
- Updates or adds entries from the new dictionary.
- Saves the merged result back to the same file.

"""


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


