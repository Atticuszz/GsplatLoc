## GSLoc-Slam
[x] clean config in base.py
1. tracking
   1. [x] normalize pcd and pose via PCA 
   2. [x] update depth_gt with a proper method 
   3. [x] loss with depth and edge and normals
   4. [x] find an early stop condition !!! -> total loss and later than 100 step
   5. [x] sync data shape and avoid too much middle vars through backward
   6. [x] total dataset eval

