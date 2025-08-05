
# 1. RLZX dataset
RLZX is a landslide dataset created by Z.F. and F.W and others, documenting 19,403 landslides triggered by Typhoon and extreme rainfall (historical record-breaking). 

The dataset consists of two main components: **RLZX-LIM**(landslide inventory, in shapefile format) and **RLZX-LDD** (landslide detection dataset). These data are intended to provide foundational data for landslide risk researches and landslide detection studies, promoting the development of research in relative fields.

The manuscript has been submitted to the "Scientific Data" journal, we will provide the download link for the dataset once accepted as soon as possible in github.

# 2. Data processing
If you want to use our RLZX-LDD dataset, you could follow the **dataloader** and the **tools** we provided in this repository.

# 3.Data download
Fu, Z., Wang, F., Ma, H., You, Q. & Feng, Y. Records of shallow landslides triggered by extreme rainfall in July 2024 in Zixing, China.
Figshare https://doi.org/10.6084/m9.fgshare.27960762 (2025).

The data is now openly accessible on figshare as the link above, **if you publish your papers using our data, please cite our work on *Scientific Data***.

# 4.Models
In this paper, we ultilized the following models to validate our datasets:  

U²-net:https://github.com/xuebinqin/U-2-Net  

Deeplabv3+:https://github.com/jfzhang95/pytorch-deeplab-xception  

Segformer:https://github.com/NVlabs/SegFormer  

MobileUNETR：https://github.com/OSUPCVLab/MobileUNETR  


Please refer to the link above to check the networks if interested.
