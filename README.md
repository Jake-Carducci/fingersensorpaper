# fingersensorpaper
Code and datasets from HAND finger sensor testing for purpose of data availability

Software used: MATLAB 2021a with Statistics and Machine Learning Toolbox (MathWorks; Natick, MA), and Solidworks 2020 (Dassault Systèmes; Vélizy-Villacoublay, France).

To replicate calibration data and figures:
1. Uncomment line 11 and comment line 12 of hdf5_read_sensors.m so active code is "filename = '06172019_s56_c.hdf5';".
2. Edit line 31 of hdf5_read_sensors.m to "isValidate = false;".
3. Run hdf5_read_sensors.m in MATLAB.

To replicate validation data and figures:
1. Uncomment line 12 and comment line 11 of hdf5_read_sensors.m so active code is "filename = '06172019_s56_v.hdf5';".
2. Edit line 31 of hdf5_read_sensors.m to "isValidate = true;".
3. Run hdf5_read_sensors.m in MATLAB.

To replicate FEA results:
1. Open "XY_Simulate.sldasm" or "strain_gage_unit.sldprt" in Solidworks and enable Simulation under Add-Ins.
2. Click the static study (i.e. "Static 1" or "X-dir_bolts_pins").
3. Hit "Run this Study" in the toolbar. Hit yes on any warnings.
4. Open the normal strain plots in the study tree panel under "Results".
