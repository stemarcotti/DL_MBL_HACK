import zarr

load_path = ['/mnt/efs/shared_data/hack/data/20230811/20230811_raw.zarr',
             '/mnt/efs/shared_data/hack/data/20230504/20230504_raw.zarr']
fov_list = [[0,1,2,3,4], [0,1,2,3]]


for i, path in enumerate(load_path):
        print(path)
        fovs = fov_list[i]
        print(f"fovs:{fovs}")
        print(i)
        f= zarr.open(path, "r")
        for k in fovs:
                print(k)
                print(f[f"fov{k}/gt"].info)