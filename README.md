## GSLoc-Slam
[ ] clean config in base.py
1. tracking
   1. [x] normalize pcd and pose via PCA ->noise
   2. [x] update depth_gt with a proper method 
   3. [x] attention layer for loss,and huber loss -> not work
   4. [x] find an early stop condition !!! -> depth loss
   5. [ ] refactor for better order
   6. [ ] total dataset eval
      1. [x] simply 
      2. add gs.add_gs method 
