# Hippocampus Volume Segmentation and Quantification using UNET
*This is part of the Udaciy AI for Healthcare Nanodegree course projects.*

### Key Words:
`3D_imaging_data`, `Segmentation`, `U-Net`, `Clinical_Integration`, `MRI`, `PyTorch`

-------
There are three sub-tasks in this project:
- `/section1/out/EDA.ipynb` consists of the instruction to analyze the NIFTI dataset and clean the dataset to remove outliers presenting in images/labels directories.
- `/section2/src/run_ml_pipeline.py` calls functions to initiate the experiment, which will build a U-net model to segment hippocampal areas from 3D MRI scans. 
- `/section3/out/inference_dcm.py` will run functions to run inference on DICOM files and generate a report (Fig.1) with visualization of segmented hippocampal area in the scan. `deploy_scripts` directory consists of the shell commands to integrate the segmentation algorithm into a simulated 
the segmentation and visualization of hippocampus using MR scans. 

<p align = "center">
<img src="./section3/out/report.png" alt="Example report for Test Volumes Study 1">
</p>
<p align = "center">
Fig.1 - Sample Report
</p>

Before running the scripts, the data should be obtained from 
[Udacity project page](https://github.com/udacity/nd320-c3-3d-imaging-starter/tree/master/data/TrainingSet)
and run the notebook `/section1/out/EDA.ipynb` to preprocess and remove outliers.
The resulted cleaned dataset need to be copied to `/section2` for the segmentation step.     

For `section3` experiment, the [TestVolumes](https://github.com/udacity/nd320-c3-3d-imaging-starter/tree/master/data/TestVolumes) dataset is used. Additionally, [Orthanc server](https://www.orthanc-server.com/download.php) and [OHIF web viewer](https://docs.ohif.org/development/getting-started.html) and [DCMTK tools](https://dcmtk.org/) need to be installed. The configuration details for Orthanc server and OHIF web viewere can be found [here](https://book.orthanc-server.com/users/lua.html). 
