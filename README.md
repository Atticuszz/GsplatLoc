## RoadMap
1. [x] experiments core env build for replica
2. [x] test grid_ds 4
   - decrease accuracy as strips up
3. [x] test gicp-small_gicp with different voxel_ds and knn for estimation covs and normals
   - knn=10 is the best as for voxel greater than 0.01
   - voxel less than 0.001 leads to extremely slow for processing
4. [x] test voxel downsample with different ratio  with open3d small_gicp ,different icps
   - no ds for cmp with my method,and set knn=20,max_co=0.1
5. [x] build depth_loss for model
6. [ ] build silhoutte_loss and color loss(or lab loss) for model
7. [ ] test for icps as 4 init params
8. [ ] test for loss optimization via build experiment for wandb select 2 rgbd
   1. step is num_iters of optimization
   2. rgb,depth,counter diff result after reconstruction
   3. total_loss plot for each step, separate for each loss then combind then compare
   4. rgb  ,SSIMLoss,MS_SSIMLoss(combind SSIMLoss+l1),psnr_loss,
   5. depth_loss first,InverseDepthSmoothnessLoss,total_variation
   6. select best loss function for optimization


normalization for depth and rgb

1. [ ] forward method ,
   - [x] DS for pcd not clean
2. [ ] and remove combination of pcd to unproject with unprojecting total that  closed 

3. [x]faiss nns can compute n pcd together on gpu for pytorch adam
   - not work ,so change to iter small_gicp kdtree

4. [x]use kornia build gicp
5. [x]GICPJacobianApprox building

1. [ ] select optimizer
   - [ ]LGBS learn slow ?

2. [x]change R as four angles
3. [ ] implement for LM optimizer
4. [ ] torch.compile for funcs
https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage

try gicp lm ,then add with gs_splatting
## Main Idea 
1. combine gicp loss into gs_splating
