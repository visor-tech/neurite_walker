set "LychnisServerPort=29738"

IF 0 EQU 1 (
    START /B ^
    .\Programs\Lychnis-1.5.8.8\Lychnis-1.5.8.8.exe ^
    Y:\SIAT_SIAT\BiGuoqiang\Macaque_Brain\RM009_2\refine_sps\221122-s100-r2000-sparse\221124-s100-r2000-sparse\231010-s100-r2000\Analysis\all-in-one\2.1.0_new\neuron#%userinput%.lyp
) ELSE (
    START /B ^
    .\Programs\Lychnis-1.5.8.8\Lychnis-1.5.8.8.exe ^
    "Y:\SIAT_SIAT\BiGuoqiang\Macaque_Brain\RM009_2\refine_sps\221122-s100-r2000-sparse\221124-s100-r2000-sparse\231010-s100-r2000\Analysis\all-in-one\export-volume-128-1.lyp"
)

