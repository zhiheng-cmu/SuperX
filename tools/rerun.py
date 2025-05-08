import rerun as rr


data_dir = "/home/zl3466/Documents/dataset/SuperX/mini"
rrd_dir = f"{data_dir}/0116_full_with_global_pcd.rrd"


rr.init(rrd_dir, recording_id="example")
