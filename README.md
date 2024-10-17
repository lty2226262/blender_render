# Installment

```
conda create -n blender_render python==3.11
conda activate blender_render
pip install bpy opencv-python imageio[ffmpeg] tqdm git+https://github.com/gdlg/simple-waymo-open-dataset-reader protobuf==3.19 open3d pyntcloud numpy==1.26.0
```

# Run
```
python build_static_map.py .......
python render.py
```