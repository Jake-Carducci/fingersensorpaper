# fingersensorpaper
Code and datasets from HAND finger sensor testing for purpose of data availability

To replicate calibration data and figures:
Uncomment line 11 and comment line 12 so active code is "filename = '06172019_s56_c.hdf5';".
Edit line 31 of hdf5_read_sensors.m to "isValidate = false;".

To replicate validation data and figures:
Uncomment line 12 and comment line 11 so active code is "filename = '06172019_s56_v.hdf5';".
Edit line 31 of hdf5_read_sensors.m to "isValidate = true;".
