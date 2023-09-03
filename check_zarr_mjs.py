import zarr

load_path = ['/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr',
             '/mnt/efs/shared_data/hack/data/20230504/20230504_raw.zarr']
fov_list = [[0,1,2,3,4], [0,1,2,3]]


# Make a function that takes in a list of paths and fovs and prints the info for each fov
# We want an assert statement that checks that the u
def check_zarr(load_path, fov_list):
        for i, path in enumerate(load_path):
                print(path)
                fovs = fov_list[i]
                print(f"fovs:{fovs}")
                print(i)
                f= zarr.open(path, "r")
                for k in fovs:
                        print(k)
                        print(f[f"fov{k}/gt"].info)
        return
        


#######################

for i, path in enumerate(load_path):
        print(path)
        fovs = fov_list[i]
        print(f"fovs:{fovs}")
        print(i)
        f= zarr.open(path, "r")
        for k in fovs:
                print(k)
                print(f[f"fov{k}/gt"].info)