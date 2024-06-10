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

7. [ ] forward method ,DS for pcd
not clean
8. [ ] and remove combination of pcd to unproject with unprojecting total that  closed 

9. faiss nns can computing n pcd together on gpu for easier with lm optimize via pytorch