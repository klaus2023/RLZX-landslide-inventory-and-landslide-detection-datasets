
# 1. RLZX dataset
RLZX is a landslide dataset created by Z.F., F.W and others, documenting 19,403 landslides triggered by Typhoon and extreme rainfall (historical record-breaking) in Zixing, Hunan Province, China. 

The dataset consists of two main components: **RLZX-LIM**(landslide inventory, in shapefile format) and **RLZX-LDD** (landslide detection dataset). These data are intended to provide foundational data for landslide risk research and automatic landslide detection studies, promoting the development of research in relative fields.

The manuscript has now been published on the "Scientific Data" journal, anyone can access the building process and description of these data by reading this paper:

**Fu, Z., Wang, F., Ma, H. et al. Records of shallow landslides triggered by extreme rainfall in July 2024 in Zixing, China. Sci Data 12, 1364 (2025). https://doi.org/10.1038/s41597-025-05670-w.**

# 2. Data processing
If you want to use our RLZX-LDD dataset, you could follow the **dataloader** and the **tools** we provided in this repository.

# 3.Data download
Fu, Z., Wang, F., Ma, H., You, Q. & Feng, Y. Records of shallow landslides triggered by extreme rainfall in July 2024 in Zixing, China.
Figshare https://doi.org/10.6084/m9.figshare.27960762 (2025).

The data is now openly accessible on figshare as the link above, **if you publish your papers using our data, please cite our work on *Scientific Data***.

# 4.Models
In this paper, we ultilized the following models to validate our datasets:  

U²-net:https://github.com/xuebinqin/U-2-Net  

Deeplabv3+:https://github.com/jfzhang95/pytorch-deeplab-xception  

Segformer:https://github.com/NVlabs/SegFormer  

MobileUNETR：https://github.com/OSUPCVLab/MobileUNETR  


Please refer to the link above to check the segmentation networks if interested.
