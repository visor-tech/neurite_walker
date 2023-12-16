@echo off

set /p userinput="Please enter a numerical ID of the neuron: "

echo "Starting Lychnis"
set "LychnisServerPort=29738"

IF 0 EQU 1 (
    REM for publish
    START /B ^
    .\Programs\Lychnis-1.5.8.8\Lychnis-1.5.8.8.exe ^
    Y:\SIAT_SIAT\BiGuoqiang\Macaque_Brain\RM009_2\refine_sps\221122-s100-r2000-sparse\221124-s100-r2000-sparse\231010-s100-r2000\Analysis\all-in-one\2.1.0_new\neuron#%userinput%.lyp
) ELSE (
    REM for test
    START /B ^
    .\Programs\Lychnis-1.5.8.8\Lychnis-1.5.8.8.exe ^
    "Y:\SIAT_SIAT\BiGuoqiang\Macaque_Brain\RM009_2\refine_sps\221122-s100-r2000-sparse\221124-s100-r2000-sparse\231010-s100-r2000\Analysis\all-in-one\export-volume-128-1.lyp"
)

echo "Starting CMIP viewer"
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