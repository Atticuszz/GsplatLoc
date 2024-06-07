## RoadMap
1. [x] experiments core env build for replica
2. [x] test grid_ds 4
   - decrease accuracy as strips up
3. [x] test gicp-small_gicp with different voxel_ds and knn for estimation covs and normals
   - knn=10 is the best as for voxel greater than 0.01
   - voxel less than 0.001 leads to extremely slow for processing
4. [ ] test voxel downsample with different ratio  with open3d small_gicp ,different icps
5. [x] build depth_loss for model
6. [ ] build silhoutte_loss and color loss(or lab loss) for model
