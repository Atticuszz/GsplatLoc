# CHANGELOG

## v0.7.3 (2024-07-27)

### Chore

* chore: remove submoudules and clean files ([`c7afa5d`](https://github.com/Atticuszz/AB_GICP/commit/c7afa5de8f5d8c2c88d4bd9e294b11a145b71cbb))

* chore: add submoudules ([`cf9a886`](https://github.com/Atticuszz/AB_GICP/commit/cf9a886ca7df556419b6e9e7c2bb1dc5ff755cac))

### Fix

* fix: add root to  system path ([`dcdd0c6`](https://github.com/Atticuszz/AB_GICP/commit/dcdd0c6af3dad9fb5cebc0fa071b88f936836f29))

## v0.7.2 (2024-07-26)

### Chore

* chore(release): bump version to v0.7.2 ([`8e355e4`](https://github.com/Atticuszz/AB_GICP/commit/8e355e4c476fc3976bd0862eda8bd1519888935c))

### Fix

* fix: remove similarity_from_cameras ([`1b08cc0`](https://github.com/Atticuszz/AB_GICP/commit/1b08cc08368277ee2db5d6bbd7c3438470a6ef0b))

## v0.7.1 (2024-07-23)

### Chore

* chore(release): bump version to v0.7.1 ([`f468a50`](https://github.com/Atticuszz/AB_GICP/commit/f468a50035404c9737d0163c1325ed68513a610f))

### Fix

* fix: total_dataset eval should omit too close points and logger wrong best_records ([`e589830`](https://github.com/Atticuszz/AB_GICP/commit/e5898300b41c5bf265aed1d577f6b70a781266ea))

## v0.7.0 (2024-07-21)

### Chore

* chore(release): bump version to v0.7.0 ([`6e72a8c`](https://github.com/Atticuszz/AB_GICP/commit/6e72a8ce2e710be851f4a675be9b609419654ae1))

* chore: format train ([`afb8592`](https://github.com/Atticuszz/AB_GICP/commit/afb8592a1a83688122b8887356308cfb19fdcda7))

* chore: format loss func ([`87ae6e5`](https://github.com/Atticuszz/AB_GICP/commit/87ae6e59782cc9633bcb44b8a8cffd442b45b630))

* chore: add gaussian-opacity-fields submodules ([`8124863`](https://github.com/Atticuszz/AB_GICP/commit/8124863dd116ffcf93a80ab8034f78d45a767d96))

### Feature

* feat: add total dataset eval simple version ([`3aca4c6`](https://github.com/Atticuszz/AB_GICP/commit/3aca4c69747f5dafa0a88389771e3b01755afa0b))

## v0.6.4 (2024-07-14)

### Chore

* chore(release): bump version to v0.6.4 ([`131ff0b`](https://github.com/Atticuszz/AB_GICP/commit/131ff0be6df7327d0c37c4e3b4e566708be93cbe))

* chore: add torch.compile and try to add early stop method ([`994a961`](https://github.com/Atticuszz/AB_GICP/commit/994a961b81dce52d268aeffa9aebad25f74a1b8c))

### Fix

* fix: normalize scale for depth and simplify compute depth_gt in Parser ([`ebff011`](https://github.com/Atticuszz/AB_GICP/commit/ebff0118781afddb7372c54acbdcf7d45f2198cd))

## v0.6.3 (2024-07-07)

### Chore

* chore(release): bump version to v0.6.3 ([`bb7f324`](https://github.com/Atticuszz/AB_GICP/commit/bb7f324964c87ad15bda3f2c03ff1325fca9f4bd))

* chore: add different camera model.py ([`b51b89a`](https://github.com/Atticuszz/AB_GICP/commit/b51b89a377dc073754bb8d1d5bd29f8b6735de1a))

### Fix

* fix: several bugs of training ([`f80cca8`](https://github.com/Atticuszz/AB_GICP/commit/f80cca86eb007831ec898043ce1eaf402f3332a5))

## v0.6.2 (2024-07-03)

### Chore

* chore(release): bump version to v0.6.2 ([`b6027d0`](https://github.com/Atticuszz/AB_GICP/commit/b6027d00b6f862fe5aeb5bfc91353f8d150b801a))

* chore: added wandb logger to monitor training loss ([`521b051`](https://github.com/Atticuszz/AB_GICP/commit/521b051308d5d42b3ba5a4af8c018dfc2f2d968f))

* chore: add early stop for train ([`15c3290`](https://github.com/Atticuszz/AB_GICP/commit/15c329064c7d749f64289f1ff73d3a10cc4d4f34))

### Fix

* fix: failed to show lr in wandb ([`378825d`](https://github.com/Atticuszz/AB_GICP/commit/378825d5e049e70897587fd82204afef68d401d4))

## v0.6.1 (2024-07-01)

### Chore

* chore(release): bump version to v0.6.1 ([`6800491`](https://github.com/Atticuszz/AB_GICP/commit/68004914b0c922877c13f435b4adabf559941b90))

* chore: add cameraopt nn.module class ([`4f3f11c`](https://github.com/Atticuszz/AB_GICP/commit/4f3f11c6625df74bae282a2013a48310aecacbab))

### Fix

* fix: correct camera model.py ([`e4fe449`](https://github.com/Atticuszz/AB_GICP/commit/e4fe4496dd734902edc5aba303f53d69bfdb74ca))

## v0.6.0 (2024-06-30)

### Chore

* chore(release): bump version to v0.6.0 ([`c3e6c21`](https://github.com/Atticuszz/AB_GICP/commit/c3e6c21d2ffe6a36941e604e5e06c66e6eaaa813))

* chore: add gsviewer and test tranning results ([`b6595f6`](https://github.com/Atticuszz/AB_GICP/commit/b6595f6f4e00bdd67fb0ff62eb68b3fc626c0e77))

### Feature

* feat: refactor gs_model and add Parser class ([`6d30853`](https://github.com/Atticuszz/AB_GICP/commit/6d308534e8c4b572589fe7437cc75eb06c0025cb))

## v0.5.0 (2024-06-27)

### Chore

* chore(release): bump version to v0.5.0 ([`6fd59ff`](https://github.com/Atticuszz/AB_GICP/commit/6fd59ffdeb8ee0a31ede7af7bea48336f26cd088))

* chore: add setter for RGBD class property ([`ab8348a`](https://github.com/Atticuszz/AB_GICP/commit/ab8348ac8a61ff56f7ca58932cb6f6f5faeecb90))

* chore: add slice func for dataloader and tested ([`2e3390a`](https://github.com/Atticuszz/AB_GICP/commit/2e3390aa9468e9ed0092545637aaec7df61ffec3))

* chore: add func docs for normalize_torch ([`4567211`](https://github.com/Atticuszz/AB_GICP/commit/45672110ff44f4f388107ea38d9f9af190be7058))

* chore: add tests of knn and normalize_torch ([`3c3ac78`](https://github.com/Atticuszz/AB_GICP/commit/3c3ac789defb3ae9edbb4ce25cf8819637329e89))

* chore: update workflows ([`44bc981`](https://github.com/Atticuszz/AB_GICP/commit/44bc9818732b6641701db306f2370cc301a30349))

* chore: add knn test ([`fd49eda`](https://github.com/Atticuszz/AB_GICP/commit/fd49edadf8e30c9c6217e994ad8014e6eda7d043))

* chore: add knn test ([`941c83e`](https://github.com/Atticuszz/AB_GICP/commit/941c83eeebe59aa889af9ee3f8e0324c86c799c2))

* chore: add knn test ([`b407caa`](https://github.com/Atticuszz/AB_GICP/commit/b407caac1f014f5d59baeb88ca3bf8becaeec990))

* chore: add gsplat submodule ([`7944f36`](https://github.com/Atticuszz/AB_GICP/commit/7944f36b4e97bfa0a77a81429df2229848149aaa))

### Feature

* feat: add gsplat trainer.py ([`920b006`](https://github.com/Atticuszz/AB_GICP/commit/920b006ce949c482e57313dcc02a54366cac44f4))

### Fix

* fix: add tests ([`9fa896f`](https://github.com/Atticuszz/AB_GICP/commit/9fa896f411622f336e1f724cc1a379357d08fb37))

## v0.4.1 (2024-06-18)

### Chore

* chore(release): bump version to v0.4.1 ([`51cb593`](https://github.com/Atticuszz/AB_GICP/commit/51cb5935e08ad9866ee93101bbf3ccdaf619b2fd))

* chore: add submodules ([`2c6b690`](https://github.com/Atticuszz/AB_GICP/commit/2c6b690b5c7de555529d610345d6b15dce69a881))

* chore: Update README.md ([`6df36a3`](https://github.com/Atticuszz/AB_GICP/commit/6df36a387f6297273e3ab3d90628cf48dd8f6350))

* chore: Update README.md ([`851a47c`](https://github.com/Atticuszz/AB_GICP/commit/851a47c82f7f8328beb747fb59896fa77caf6136))

* chore: run pre-commits ([`47e7df4`](https://github.com/Atticuszz/AB_GICP/commit/47e7df4c35d293e9ac3e9bc89d2573adbdcd7bb8))

* chore: update README.md ([`c00503d`](https://github.com/Atticuszz/AB_GICP/commit/c00503da4f9677ffb937ecc59baa26eb1fe85a61))

* chore: update README.md ([`fd9a040`](https://github.com/Atticuszz/AB_GICP/commit/fd9a04031628997244dfe7ff0dd04fc55b5d9be5))

### Fix

* fix: small_gicp batch_knn api and add submodules ([`f80c0eb`](https://github.com/Atticuszz/AB_GICP/commit/f80c0eb90db7c27bf9c47b62a4a599fd4e24b897))

## v0.4.0 (2024-06-17)

### Chore

* chore(release): bump version to v0.4.0 ([`e038e9f`](https://github.com/Atticuszz/AB_GICP/commit/e038e9f10d5cb8aad4b458c67e83360f7b6df6d5))

### Feature

* feat: add knnsearch keops and knn tests ([`2f57ae7`](https://github.com/Atticuszz/AB_GICP/commit/2f57ae7c50db498055034efb06f673965bcc180b))

## v0.3.0 (2024-06-16)

### Chore

* chore(release): bump version to v0.3.0 ([`6b5120c`](https://github.com/Atticuszz/AB_GICP/commit/6b5120cf51c6eeb82c3dd2fb3f1a15bd820303a5))

* chore: GICPJacobianApprox building ([`bd3badc`](https://github.com/Atticuszz/AB_GICP/commit/bd3badcac675c3de7ec4413efab00cb595c1070f))

* chore: Update README.md ([`47faf50`](https://github.com/Atticuszz/AB_GICP/commit/47faf5002d65fdfd787b0d0dc56bb7ac7d29b5d7))

* chore: Update README.md ([`55c5384`](https://github.com/Atticuszz/AB_GICP/commit/55c5384285df558864b24ce91e434e5078600f9a))

### Feature

* feat: GICPJacobianApprox building finished ([`1e1b896`](https://github.com/Atticuszz/AB_GICP/commit/1e1b8965c975a0e77611377d37ca3a8625115f0a))

### Fix

* fix: reshape pcd from h,w,4 to h*w,4 ([`d855924`](https://github.com/Atticuszz/AB_GICP/commit/d855924d406b1cf67e8bef83602a0a27f0544adb))

## v0.2.3 (2024-06-10)

### Chore

* chore(release): bump version to v0.2.3 ([`152f026`](https://github.com/Atticuszz/AB_GICP/commit/152f02621862c1af6cddefff576ee05441475020))

### Fix

* fix: Update gemoetry.py ([`20a8122`](https://github.com/Atticuszz/AB_GICP/commit/20a8122bf44a81975c856750ae92a5d1017284cb))

## v0.2.2 (2024-06-09)

### Chore

* chore(release): bump version to v0.2.2 ([`a4e2f70`](https://github.com/Atticuszz/AB_GICP/commit/a4e2f700597d0401c90bc6f3aba0b09b7b4e964a))

### Fix

* fix: model.py Merge point clouds using a differentiable method ([`c9082db`](https://github.com/Atticuszz/AB_GICP/commit/c9082db821ef56042e7e14c43c3ea15c0b43e23c))

## v0.2.1 (2024-06-09)

### Chore

* chore(release): bump version to v0.2.1 ([`4f48c56`](https://github.com/Atticuszz/AB_GICP/commit/4f48c5640173880a0f584125eb86f1c467f4c674))

### Fix

* fix: make unproject_depth func differentiable ([`11aa2a3`](https://github.com/Atticuszz/AB_GICP/commit/11aa2a3ad73d9db41418b58406397ae3099e509f))

## v0.2.0 (2024-06-09)

### Chore

* chore(release): bump version to v0.2.0 ([`ee71281`](https://github.com/Atticuszz/AB_GICP/commit/ee712815ab0e2ee8cc4c68dc40f001b154027646))

* chore: reshape model pys ([`6e263ec`](https://github.com/Atticuszz/AB_GICP/commit/6e263ecdd3282f9f350171a7c96cfc0a033efe86))

* chore: update README.md ([`e19e9fd`](https://github.com/Atticuszz/AB_GICP/commit/e19e9fd8ac0b21b61918d04100f42aa882eb5660))

* chore: add counter loss and lbfgs optimizer ([`0441ab7`](https://github.com/Atticuszz/AB_GICP/commit/0441ab7430fddaa9891a1fddd52ee79ed87e2bfb))

### Feature

* feat: add training model vis wandb loger ([`848eb89`](https://github.com/Atticuszz/AB_GICP/commit/848eb89ca2833b02f58ab0cc5d3add998915c1a2))

### Unknown

* Update README.md ([`2fdc295`](https://github.com/Atticuszz/AB_GICP/commit/2fdc29576fe1b7e0814539ffbf3ecc47b6dfe594))

## v0.1.0 (2024-06-07)

### Chore

* chore(release): bump version to v0.1.0 ([`4172712`](https://github.com/Atticuszz/AB_GICP/commit/417271229f6a570b72e713101f64f25a0068ce8c))

* chore: finished depth_loss ([`80d6414`](https://github.com/Atticuszz/AB_GICP/commit/80d6414dce07fc56a8b42585619ec50960b9f1ec))

* chore: add depth_loss.py ([`4421f40`](https://github.com/Atticuszz/AB_GICP/commit/4421f40cab006450fca4d8eb372704bc19ed194b))

* chore: format code ([`588687e`](https://github.com/Atticuszz/AB_GICP/commit/588687e074518d443f74c04c57716711580bcc1f))

* chore: adding lm ([`9084fa2`](https://github.com/Atticuszz/AB_GICP/commit/9084fa28814676d6455f965af15233d44f9eab77))

* chore: mass ([`1a092b9`](https://github.com/Atticuszz/AB_GICP/commit/1a092b97c5cd1d0821298e51ba1ef745a48d63b9))

* chore: build pointclouds class ([`61174f6`](https://github.com/Atticuszz/AB_GICP/commit/61174f6073b5e727ccc1049cc0ce590e6b162fc9))

* chore: update REAMDME.md ([`0e3dede`](https://github.com/Atticuszz/AB_GICP/commit/0e3dede53fb2a19201df88c04b40a2ba701890be))

* chore: finished knn test,select small_gicp ([`2afd883`](https://github.com/Atticuszz/AB_GICP/commit/2afd883117a1aed9633fe31a1ddb4bb63215c717))

### Feature

* feat: add color icp and finished the experiments ([`eb68c7f`](https://github.com/Atticuszz/AB_GICP/commit/eb68c7f0a645233241f3b9679fab82f5f524b19e))

* feat: add o3d as gicp backend ([`04cad3f`](https://github.com/Atticuszz/AB_GICP/commit/04cad3fab7493fdb680ddc39f969befd7cb79470))

* feat: adding gicp loss ([`e1b9dfe`](https://github.com/Atticuszz/AB_GICP/commit/e1b9dfee6b3858894050b81a32ff2b09485279ad))

* feat: finished experiment class ([`b7d32c2`](https://github.com/Atticuszz/AB_GICP/commit/b7d32c22a8c8aba402d603eea64db5eda3693cbe))

### Fix

* fix: device selection in pytorch ([`4bcfd1e`](https://github.com/Atticuszz/AB_GICP/commit/4bcfd1ee110ab6b27f03e7143a4ae8c4634fe774))

* fix: config experiment envs ([`24c1560`](https://github.com/Atticuszz/AB_GICP/commit/24c15608e9ee623b4f004a53ec8415ec6dcf68f7))

* fix: config error in grip_downsample ([`160830c`](https://github.com/Atticuszz/AB_GICP/commit/160830cf843f93b97e7886365da816a962c60051))

## v0.0.1 (2024-05-24)

### Chore

* chore(release): bump version to v0.0.1 ([`e3e7ee6`](https://github.com/Atticuszz/AB_GICP/commit/e3e7ee62a53566075dcd1680da4e79fdc9cf47c0))

### Fix

* fix: ci ([`8348140`](https://github.com/Atticuszz/AB_GICP/commit/83481401830ebef87ce4bfd3819955afb4532d64))

### Unknown

* finished eval func ([`c86a5b8`](https://github.com/Atticuszz/AB_GICP/commit/c86a5b875b8b1f0bc64514a18660c5fc1d67df06))

* finished eval ([`bd32781`](https://github.com/Atticuszz/AB_GICP/commit/bd32781c588fbe4de5be31964f64217551c05914))

* init build env ([`ee3206e`](https://github.com/Atticuszz/AB_GICP/commit/ee3206e638fe8ed6adaa8b1e075919e5264a6c82))

* Update pyproject.toml ([`e268f5a`](https://github.com/Atticuszz/AB_GICP/commit/e268f5adfef705449bd558aac8ed68cb19005c31))

* Delete poetry_scripts.py ([`44244c0`](https://github.com/Atticuszz/AB_GICP/commit/44244c0f9b275a98147ba648d47df0146d7e681a))

* Merge pull request #1 from Atticuszz/dependabot/pip/pytest-cov-5.0.0

⬆ Bump pytest-cov from 4.1.0 to 5.0.0 ([`f2fe1c4`](https://github.com/Atticuszz/AB_GICP/commit/f2fe1c4482cd03ba1f2f90172c62f19a51524406))

* ⬆ Bump pytest-cov from 4.1.0 to 5.0.0

Bumps [pytest-cov](https://github.com/pytest-dev/pytest-cov) from 4.1.0 to 5.0.0.
- [Changelog](https://github.com/pytest-dev/pytest-cov/blob/master/CHANGELOG.rst)
- [Commits](https://github.com/pytest-dev/pytest-cov/compare/v4.1.0...v5.0.0)

---
updated-dependencies:
- dependency-name: pytest-cov
  dependency-type: direct:development
  update-type: version-update:semver-major
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`ec0136c`](https://github.com/Atticuszz/AB_GICP/commit/ec0136c8ac4684579134609c0e70d556114cc798))

* Initial commit ([`fc7eef8`](https://github.com/Atticuszz/AB_GICP/commit/fc7eef83a8ff7f8eb88ec0332ffbe2b8909bd699))
