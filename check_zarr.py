import os
import zarr
import glob

directory_path = '/mnt/efs/shared_data/hack/data/'

# Use glob to recursively find all .zarr files
all_files = glob.glob(os.path.join(directory_path, '**', '*.zarr'), recursive=True)

for path in all_files:
    try:
        f = zarr.open(path, "r")
    except ValueError:
        print(f"Error: Unable to open zarr file at {path}")
        continue

    print(f"\nChecking zarr file: {path}")

    # Dynamically determine fov numbers by examining keys in the zarr file.
    fovs = set([key.split('/')[0] for key in f.keys() if "fov" in key])

    for fov in sorted(fovs):
        for key_name in ["gt", "raw", "fg_mask"]:
            full_key = f"{fov}/{key_name}"
            try:
                dtype = f[full_key].dtype
                print(f"FOV: {fov.replace('fov', '')}, Key: {key_name}, Dtype: {dtype}")
            except KeyError:
                print(f"Error: Key '{full_key}' not found in {path}")
