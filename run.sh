# init shape, if shape is fixed, we could not run this code
# python demo_shape_init.py -c configs/init_smpl_rgbd.yaml

# pose tracking
python demo.py -c configs/fit_smpl_rgbd.yaml --init_model
python demo.py -c configs/fit_smpl_rgbd.yaml 
# python -m apps.run --config configs/fit_smpl_rgbd.yaml --gpus "0"