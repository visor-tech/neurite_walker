@echo off

set /p userinput="Please enter a numerical ID of the neuron: "

set MPLBACKEND=qtagg

.\Programs\python-3.11.6-embed-amd64\python.exe ^
%cd%/code/neurite_walker/neu_walk.py ^
--zarr_dir %cd%/dataset/RM009/blk128_neu210_sps231209.zarr ^
--cmip_dir %cd%/dataset/RM009/cmip_swc210n_res1 ^
--res 1 ^
--view_length 2000/3 ^
--filter "(branch_depth(processes)<=7) & (path_length_to_root(end_point(processes))>10000)" ^
--viewer lychnis ^
--view %cd%/dataset/RM009/swc210n_largest/neuron#%userinput%.lyp.swc