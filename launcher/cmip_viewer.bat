@echo off

set /p userinput="Please enter a numerical ID of the neuron: "

set MPLBACKEND=qtagg

.\Programs\python-3.11.6-embed-amd64\python.exe %cd%/code/neurite_walker/neu_walk.py --zarr_dir %cd%/dataset/RM009/blk128_neu172_sps231010.zarr --cmip_dir %cd%/dataset/RM009/cmip_swc172_res1 --res 1 --filter "(branch_depth(processes)<=3) & (path_length_to_root(end_point(processes))>10000)" --view %cd%/dataset/RM009/swc1.7.2_largest/neuron#%userinput%.lyp.swc
