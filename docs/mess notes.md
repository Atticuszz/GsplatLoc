GSplatLoc，这是我的一个基于3d高斯显示场景表示法的重投影误差方法，你给我取几个标题并且一个标题后面的介绍就像这样SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM，Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting，重投影3d高斯姿态估计超高精度估计方法，旋转误差几乎为0


侧重点应该是重投影和姿态优化方法


**GSplatLoc: Ultra-Precise Pose Optimization via 3D Gaussian Reprojection** GSplatLoc offers an ultra-precise approach to pose optimization using 3D Gaussian reprojection. By meticulously minimizing reprojection errors, this method achieves exceptional accuracy in 3D pose estimation, essential for detailed and realistic scene reconstruction.


Abstract. We present a dense simultaneous localization and mapping (SLAM) method that uses 3D Gaussians as a scene representation. Our approach enables interactive-time reconstruction and photo-realistic rendering from real-world single-camera RGBD videos. To this end, we propose a novel effective strategy for seeding new Gaussians for newly explored areas and their effective online optimization that is independent of the scene size and thus scalable to larger scenes. This is achieved by organizing the scene into sub-maps which are independently optimized and do not need to be kept in memory. We further accomplish frame-tomodel camera tracking by minimizing photometric and geometric losses between the input and rendered frames. The Gaussian representation allows for high-quality photo-realistic real-time rendering of real-world scenes. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance in mapping, tracking, and rendering compared to existing neural dense SLAM methods.，Dense simultaneous localization and mapping (SLAM) is crucial for robotics and augmented reality applications. However, current methods are often hampered by the nonvolumetric or implicit way they represent a scene. This work introduces SplaTAM, an approach that, for the first time, leverages explicit volumetric representations, i.e., 3D Gaussians, to enable high-fidelity reconstruction from a single unposed RGB-D camera, surpassing the capabilities of existing methods. SplaTAM employs a simple online tracking and mapping system tailored to the underlying Gaussian representation. It utilizes a silhouette mask to elegantly capture the presence of scene density. This combination enables several benefits over prior representations, including fast rendering and dense optimization, quickly determining if areas have been previously mapped, and structured map expansion by adding more Gaussians. Extensive experiments show that SplaTAM achieves up to 2× superior performance in camera pose estimation, map construction, and novel-view synthesis over existing methods, paving the way for more immersive high-fidelity SLAM applications.，模仿他们的风格，给我写一个abstract，标题我选用了GSplatLoc: Ultra-Precise Pose Optimization via 3D Gaussian Reprojection，我们提出来一种基于3d高斯显示体积表示时的一种高精度姿态优化方法，3d高斯的重投影，适用于没有姿态的RGB-D相机，主要针对于rgb-d数据，主要利用了通过观察不同视角的现有3d高斯和实际拍摄的深度图的进行优化当前姿态，实现了旋转误差几乎为0，平移误差为0.01mm以内，并且和现有对rgb0d数据姿态估计的点云对其算法做了详细的比较，姿态误差实现百倍提升




然后你开始给我写一些公式，深度图的定义，2.2 Depth Compositing of Gaussians We directly follow the tile sorting method introduced by [Kerbl et al., 2023], which bins the 2D Gaussians into 16 × 16 tiles and sorts them per tile by depth. For each Gaussian, we compute the axis-aligned bounding box around the 99% confidence ellipse of each 2D projected covariance (3 sigma), and include it in a tile bin if its bounding box intersects with the tile. We then apply the tile sorting algorithm as presented in Appendix C of [Kerbl et al., 2023] to get a list of Gaussians sorted by depth for each tile. We then rasterize the sorted Gaussians within each tile. For a color at a pixel i, let n index the N Gaussians involved in that pixel. Ci = ∑ n≤N cn · αn · Tn , where Tn = ∏ m<n (1 − αm ). (7) We compute α with the 2D covariance Σ′ ∈R2× 2 and opacity parameters: αn = on · exp(−σn ), σn = 1 2 ∆⊤ n Σ ′−1∆ n , where ∆ ∈R2 and is the offset between the pixel center and the 2D Gaussian center μ′ ∈R2. We compute Tn online as we iterate through the Gaussians front to back，帮我先定义一下深度信息的数学公式，根据图片内容，我需要定义重投影采用的方法是什么？以及损失是如何设计的，采用一部分深度l1损失，一部分深度轮廓损失，



localization for Dense simultaneous localization and mapping (SLAM)中的定位方法部分

# abstract

We present GSplatLoc, an innovative pose estimation method for RGB-D cameras that employs a volumetric representation of 3D Gaussians. This approach facilitates precise pose estimation by minimizing the loss based on the reprojection of 3D Gaussians from real depth maps captured from the estimated pose. Our method attains rotational errors close to zero and translational errors within 0.01mm, representing a substantial advancement in pose accuracy over existing point cloud registration algorithms, as well as explicit volumetric and implicit neural representation-based SLAM methods. Comprehensive evaluations demonstrate that GSplatLoc significantly improves pose estimation accuracy, which contributes to increased robustness and fidelity in real-time 3D scene reconstruction, setting a new standard for localization techniques in dense mapping SLAM.

这种定位方法既可以集成到基于高斯的slam，也可以集成到gs的navigation中，提供了

# Relate works
和orb特征点相关的有[@huangPhotoSLAMRealtimeSimultaneous2024]这个过程的目标是最小化匹配帧的2D几何关键点和3D点之间的重新投影误差。
[@huCGSLAMEfficientDense2024]，跟踪损失由颜色损失和几何损失组成，
[@matsukiGaussianSplattingSlam2024]优化目标函数结合了光度残差和几何残差,
# Methods


这是一段参考文本，但是描述的不是我的方法，需要修改成我的方法，我会在对应的句子后面添加应该修改的内容
**Problem formulation:** Our goal is to estimate the 6-DoF pose (R , t) ∈ SE(3) of a query image I q，这里查询的应该是是深度图像, where R is a rotation matrix and t is a translation vector in the camera frame.姿态确实是深度相机的姿态 We are given a 3D representation of the environment, such as a sparse or dense 3D point cloud我们这里不是点云了应该是3D Gaussians,是3D高斯 { P i } and posed reference images { I k }和有姿态的参考深度图 , collectively called the reference data.




In the context of depth projection and rasterization for Gaussian splatting, the process involves several mathematical transformations to project 3D Gaussians onto a 2D image plane. Here’s a detailed explanation of the implementation based on the provided code and mathematical principles:

1. **Camera Transformations**:
    - **Extrinsics $T_{cw}$**: This matrix transforms points from the world coordinate space to the camera coordinate space. It is defined as:
      $$
      T_{cw} = \begin{bmatrix} R_{cw} & t_{cw} \\ 0 & 1 \end{bmatrix} \in SE(3)
      $$
      where $R_{cw}$ is the rotation matrix, and $t_{cw}$ is the translation vector.

    - **Projection Matrix $P$**: This matrix transforms points from camera space to normalized device coordinates (ND). It is defined as:
      $$
      P = \begin{bmatrix} \frac{2f_x}{w} & 0 & 0 & 0 \\ 0 & \frac{2f_y}{h} & 0 & 0 \\ 0 & 0 & \frac{f+n}{f-n} & -\frac{2fn}{f-n} \\ 0 & 0 & 1 & 0 \end{bmatrix}
      $$
      where $w$ and $h$ are the width and height of the output image, and $n$ and $f$ are the near and far clipping planes.

2. **Projection of 3D Gaussians**:
    - The 3D mean $\mu$ of the Gaussian is projected into pixel space:
      $$
      t = T_{cw} \begin{bmatrix} \mu \\ 1 \end{bmatrix}, \quad t' = P t, \quad \mu' = \begin{bmatrix} \frac{w \cdot t'_x / t'_w + 1}{2} + c_x \\ \frac{h \cdot t'_y / t'_w + 1}{2} + c_y \end{bmatrix}
      $$

3. **Covariance Transformation**:
    - The 3D Gaussian covariance $\Sigma$ is approximated in the 2D pixel space using the Jacobian $J$:
      $$
      J = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \end{bmatrix}
      $$
      The 2D covariance $\Sigma'$ is then:
      $$
      \Sigma' = J R_{cw} \Sigma R_{cw}^T J^T
      $$

4. **Depth Compositing**:
    - Gaussians are sorted by depth and composited from front to back. The colour $C_i$ at pixel $i$ is computed as:
      $$
      C_i = \sum_{n \leq N} c_n \cdot \alpha_n \cdot T_n, \quad T_n = \prod_{m<n} (1 - \alpha_m)
      $$
    - The opacity $\alpha$ is computed as:
      $$
      \alpha_n = o_n \cdot \exp(-\sigma_n), \quad \sigma_n = \frac{1}{2} \Delta_n^T \Sigma'^{-1} \Delta_n
      $$
      where $\Delta$ is the offset between the pixel center and the 2D Gaussian center $\mu'$.

#### Gaussian Model Explanation
The Gaussian model in the provided code involves several parameters and transformations:

1. **3D Means and Covariance**:
    - The 3D means ($\mu$) are derived from the point cloud data.
    - The 3D covariance ($\Sigma$) is parameterized by scale ($s$) and rotation quaternion ($q$). The covariance is computed as:
      $$
      \Sigma = R S S^T R^T, \quad R = \text{rotation matrix from quaternion } q, \quad S = \text{diag}(s)
      $$



To generate the depth map, we employ a front-to-back compositing strategy. For each pixel $p$, its depth value $d_p$ is computed as:
$$d_p = \sum_i w_i z_i$$
where $z_i$ represents the depth of the $i$-th Gaussian's mean, and $w_i$ is the weight derived from the 2D Gaussian distribution:
$$w_i = \exp\left(-\frac{1}{2}(x_p - \mu_{I,i})^T \Sigma_{I,i}^{-1} (x_p - \mu_{I,i})\right)$$
Here, $x_p$ is the 2D coordinate of pixel $p$, $\mu_{I,i}$ and $\Sigma_{I,i}$ denote the projected mean and covariance of the $i$-th Gaussian.

This approach enables efficient depth map generation by leveraging the dense point cloud captured by the depth camera, without requiring colour information.

## Depth Compositing 


Depth at a pixel $i$ is represented by combining contributions from multiple Gaussian elements, each associated with a certain depth and confidence. Depth $D_i$ can be expressed as[@kerbl3dGaussianSplatting2023]:
$$ 
D_i = \frac{\sum_{n \leq N} d_n \cdot c_n \cdot \alpha_n \cdot T_n}{\sum_{n \leq N} c_n \cdot \alpha_n \cdot T_n} 
$$
$d_n$ is the depth value from the $n$-th Gaussian, $c_n$ is the confidence or weight of the $n$-th Gaussian,$\alpha_n$ is the opacity calculated from Gaussian parameters, $T_n$ is the product of transparencies from all Gaussians in front of the $n$-th Gaussian.

The reprojection method utilizes the alignment of 2D Gaussian projections with observed depth data from an RGB-D camera. This involves adjusting the parameters of the Gaussians to minimize the discrepancy between the projected depth and the observed depth. The offset $\Delta_n$ and the covariance matrix $\Sigma'$ are crucial for calculating the Gaussian weights $\alpha_n$ and their impact on reprojection accuracy.


深度生成
[gsplat/gsplat/rendering.py at main · nerfstudio-project/gsplat · GitHub](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/rendering.py)
```python
if render_mode in ["ED", "RGB+ED"]: 
	# normalize the accumulated depth to get the expected depth 
	render_colors = torch.cat( [ render_colors[..., :-1],
								render_colors[..., -1:] / render_alphas.clamp(min=1e-10), ], dim=-1, 
							)
```
我是用的是ED方式，也就是说，



## Camera Tracking




这里你要写一个最小化目标，并且我用的是adam优化器，解释优化方法，
```python
@dataclass(frozen=True)
class CameraConfig:
    trans_lr: float = 1e-3
    quat_lr: float = 5 * 1e-4
    quat_opt_reg: float = 1e-3
    trans_opt_reg: float = 1e-3
    def _create_optimizers(self) -> list[Optimizer]:
        params = [
            # name, value, lr
            # ("means3d", self.means3d, self.lr_means3d),
            ("quat", self.quaternion_cur, self.config.quat_lr),
            ("trans", self.translation_cur, self.config.trans_lr),
        ]
        optimizers = [
            Adam(
                [
                    {
                        "params": param,
                        "lr": lr,
                        "name": name,
                    }
                ],
                weight_decay=(
                    self.config.quat_opt_reg
                    if name == "quat"
                    else self.config.trans_opt_reg
                ),
            )
            for name, param, lr in params
        ]
        return optimizers
```
,并且我是定义了weight_decay，,这个你也要体现在优化目标中，总损失应该是定义了 一个正则项的


##  Localization pipeline

在这个定义下，我又产生了新的段落，来描述初始化高斯的过程，
## Localization pipeline


We initialize these Gaussians from a point cloud, where each point corresponds to a Gaussian's mean $\boldsymbol{\mu}_i$.
Unlike traditional 3D reconstruction methods[@kerbl3dGaussianSplatting2023] that often rely on structure-from-motion techniques[@schonbergerStructurefrommotionRevisited2016], our approach is tailored for direct point cloud input, offering greater flexibility and efficiency in various 3D data scenarios. For the initial parameterization, we set $o_i = 1$ for all Gaussians to ensure full opacity. The scale $\mathbf{s}_i \in \mathbb{R}^3$ of each Gaussian is initialized based on the local point density, allowing our model to adaptively adjust to varying point cloud densities:

$$\mathbf{s}_i = (\sigma_i, \sigma_i, \sigma_i), \text{ where } \sigma_i = \sqrt{\frac{1}{3}\sum_{j=1}^3 d_{ij}^2}$$

Here, $d_{ij}$ is the distance to the $j$-th nearest neighbour of point $i$. In practice, we calculate this using the k-nearest neighbours algorithm with $k=4$, excluding the point itself. This isotropic initialization ensures a balanced initial representation of the local geometry.

Initially, we set $\mathbf{q}_i = (1, 0, 0, 0)$ for all Gaussians, corresponding to no rotation. This initialization strategy provides a neutral starting point, allowing subsequent optimization processes to refine the orientations as needed.


就像这篇论文一样，4. Localization pipeline PixLoc can be a competitive standalone localization module when coupled with image retrieval, but can also refine poses obtained by previous approaches. It only requires a 3D model and a coarse initial pose, which we now discuss. Initialization: How accurate the initial pose should be depends on the basin of convergence of the alignment. Features from a deep CNN with a large receptive field ensure a large basin (Figure 5). To further increase it, we apply PixLoc to image pyramids, starting at the lowest resolution, yielding coarsest feature maps of size W=16. To keep the pipeline simple, we select the initial pose as the pose of the first reference image returned by image retrieval. This results in a good convergence in most scenarios. When retrieval is not sufficiently robust and returns an incorrect location, as in the most challenging conditions, one could improve the performance by reranking using covisiblity clustering [70,73] or pose verification with sparse [72,96] or dense matching [82]. 3D structure: For simplicity and unless mentioned, for both training and evaluation, we use sparse SfM models triangulated from posed reference images using hloc [69,70] and COLMAP [77,79]. Given a subset of reference images, e.g. top-5 retrieved, we gather all the 3D points that they observe, extract multilevel features at their 2D observations, and average them based on their confidence.，然后这是我的参考我已经写完的方法论部分，

**Problem formulation**: Our objective is to estimate the 6-DoF pose $(R, t) \in SE(3)$ of a query depth image $D_q$, where $R$ is the rotation matrix and $t$ is the translation vector in the camera coordinate system. Given a 3D representation of the environment in the form of 3D Gaussians, let $\mathcal{G} = \{G_i\}_{i=1}^N$ denote a set of $N$ 3D Gaussians, and posed reference depth images $\{D_k\}$, which together constitute the reference data.


## Gaussian Splatting


Each Gaussian $G_i$ is characterized by its 3D mean $\boldsymbol{\mu}_i \in \mathbb{R}^3$, 3D covariance matrix $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3\times3}$, opacity $o_i \in \mathbb{R}$, and scale $\mathbf{s}_i \in \mathbb{R}^3$. To represent the orientation of each Gaussian, we use a rotation quaternion $\mathbf{q}_i \in \mathbb{R}^4$.

The 3D covariance matrix $\boldsymbol{\Sigma}_i$ is then parameterized using $\mathbf{s}_i$ and $\mathbf{q}_i$:

$$\boldsymbol{\Sigma}_i = R(\mathbf{q}_i) S(\mathbf{s}_i) S(\mathbf{s}_i)^T R(\mathbf{q}_i)^T$$

where $R(\mathbf{q}_i)$ is the rotation matrix derived from $\mathbf{q}_i$, and $S(\mathbf{s}_i) = \text{diag}(\mathbf{s}_i)$ is a diagonal matrix of scales.

To project these 3D Gaussians onto a 2D image plane, we follow the approach described by [@kerbl3dGaussianSplatting2023]. The projection of the 3D mean $\boldsymbol{\mu}_i$ to the 2D image plane is given by:

$$\boldsymbol{\mu}_{I,i} = \pi(P(T_{wc} \boldsymbol{\mu}_{i,\text{homogeneous}}))$$

where $T_{wc} \in SE(3)$ is the world-to-camera transformation, $P \in \mathbb{R}^{4 \times 4}$ is the projection matrix [@yeMathematicalSupplementTexttt2023], and $\pi: \mathbb{R}^4 \rightarrow \mathbb{R}^2$ maps to pixel coordinates.

The 2D covariance $\boldsymbol{\Sigma}_{I,i} \in \mathbb{R}^{2\times2}$ of the projected Gaussian is derived as:

$$\boldsymbol{\Sigma}_{I,i} = J R_{wc} \boldsymbol{\Sigma}_i R_{wc}^T J^T$$

where $R_{wc}$ represents the rotation component of $T_{wc}$, and $J$ is the affine transform as described by [@zwickerEWASplatting2002].


## Depth Compositing



For depth map generation, we employ a front-to-back compositing scheme, which allows for accurate depth estimation and edge alignment. Let $d_n$ represent the depth value associated with the $n$-th Gaussian, which is the z-coordinate of the Gaussian's mean in the camera coordinate system. The depth $D(p)$ at pixel $p$ is computed as [@kerbl3dGaussianSplatting2023]:

$$D(p) = \sum_{n \leq N} d_n \cdot \alpha_n \cdot T_n, \quad \text{where } T_n = \prod_{m<n} (1 - \alpha_m)$$

Here, $\alpha_n$ represents the opacity of the $n$-th Gaussian at pixel $p$, computed as:

$$\alpha_n = o_n \cdot \exp(-\sigma_n), \quad \sigma_n = \frac{1}{2} \Delta_n^T \boldsymbol{\Sigma}_I^{-1} \Delta_n$$

where $\Delta_n$ is the offset between the pixel center and the 2D Gaussian center $\boldsymbol{\mu}_I$, and $o_n$ is the opacity parameter of the Gaussian. $T_n$ denotes the cumulative transparency product of all Gaussians preceding $n$, accounting for the occlusion effects of previous Gaussians.

To ensure consistent representation across the image, we normalize the depth values. First, we calculate the total accumulated opacity $\alpha(p)$ for each pixel:

$$\alpha(p) = \sum_{n \leq N} \alpha_n \cdot T_n$$

The normalized depth $\text{Norm}_D(p)$ is then defined as:

$$\text{Norm}_D(p) = \frac{D(p)}{\alpha(p)}$$

This normalization process ensures that the depth values are properly scaled and comparable across different regions of the image, regardless of the varying densities of Gaussians in the scene. By projecting 3D Gaussians onto the 2D image plane and computing normalized depth values, we can effectively generate depth maps that accurately represent the 3D structure of the scene while maintaining consistency across different viewing conditions.

## Camera Pose



We define the camera pose as

$$
 \mathbf{T}_{cw} = \begin{pmatrix} \mathbf{R}_{cw} & \mathbf{t}_{cw} \\ \mathbf{0} & 1 \end{pmatrix} \in SE(3)
$$

where $\mathbf{T}_{cw}$ represents the camera-to-world transformation matrix. Notably, we parameterize the rotation $\mathbf{R}_{cw} \in SO(3)$ using a quaternion $\mathbf{q}_{cw}$. This choice of parameterization is motivated by several key advantages that quaternions offer in the context of camera pose estimation and optimization. Quaternions provide a compact and efficient representation, requiring only four parameters, while maintaining numerical stability and avoiding singularities such as gimbal lock. Their continuous and non-redundant nature is particularly advantageous for gradient-based optimization algorithms, allowing for unconstrained optimization and simplifying the optimization landscape.

## Optimization
Based on these considerations, we design our optimization variables to separately optimize the normalized quaternion and the translation. The loss function is designed to ensure accurate depth estimations and edge alignment, incorporating both depth magnitude and contour accuracy. It can be defined as:

$$ 
L = \lambda_1 \cdot L_{\text{depth}} + \lambda_2 \cdot L_{\text{contour}} 
$$

where $L_{\text{depth}}$ represents the L1 loss for depth accuracy, and $L_{\text{contour}}$ focuses on the alignment of depth contours or edges. Specifically:

$$
L_{\text{depth}} = \sum_{i \in M} |D_i^{\text{predicted}} - D_i^{\text{observed}}|
$$

$$
L_{\text{contour}} = \sum_{j \in M} |\nabla D_j^{\text{predicted}} - \nabla D_j^{\text{observed}}|
$$

Here, $M$ denotes the reprojection mask, indicating which pixels are valid for reprojection. Both $L_{\text{depth}}$ and $L_{\text{contour}}$ are computed only over the masked regions. $\lambda_1$ and $\lambda_2$ are weights that balance the two parts of the loss function, tailored to the specific requirements of the application.

The optimization objective can be formulated as:

$$
\min_{\mathbf{q}_{cw}, \mathbf{t}_{cw}} L + \lambda_q \|\mathbf{q}_{cw}\|_2^2 + \lambda_t \|\mathbf{t}_{cw}\|_2^2
$$

where $\lambda_q$ and $\lambda_t$ are regularization terms for the quaternion and translation parameters, respectively.

We employ the Adam optimizer for both quaternion and translation optimization, with different learning rates and weight decay values for each. The learning rates are set to $5 × 10^-4$ for quaternion optimization and $10^-3$ for translation optimization, based on experimental results. The weight decay values are set to $10^-3$ for both quaternion and translation parameters, serving as regularization to prevent overfitting.
你需要给方法论部分添加一个Localization pipeline，
第一段应该是说定位pipelind的精确的简介描述，我给出我的描述，你不需要全部写出来，提供给你参考，用给定姿态的深度图生成了gs，给定查询深度图的姿态和深度数据本身，然后进行优化求解，我的描述比较口语化，但你写的必须是符合书面论文要求的，计算机顶会论文的标准 
第二段落 下面这个段落是高斯初始化的详细描述We initialize these Gaussians from a point cloud, where each point corresponds to a Gaussian's mean $\boldsymbol{\mu}_i$.
Unlike traditional 3D reconstruction methods[@kerbl3dGaussianSplatting2023] that often rely on structure-from-motion techniques[@schonbergerStructurefrommotionRevisited2016], our approach is tailored for direct point cloud input, offering greater flexibility and efficiency in various 3D data scenarios. For the initial parameterization, we set $o_i = 1$ for all Gaussians to ensure full opacity. The scale $\mathbf{s}_i \in \mathbb{R}^3$ of each Gaussian is initialized based on the local point density, allowing our model to adaptively adjust to varying point cloud densities:

$$\mathbf{s}_i = (\sigma_i, \sigma_i, \sigma_i), \text{ where } \sigma_i = \sqrt{\frac{1}{3}\sum_{j=1}^3 d_{ij}^2}$$

Here, $d_{ij}$ is the distance to the $j$-th nearest neighbour of point $i$. In practice, we calculate this using the k-nearest neighbours algorithm with $k=4$, excluding the point itself. This isotropic initialization ensures a balanced initial representation of the local geometry.

Initially, we set $\mathbf{q}_i = (1, 0, 0, 0)$ for all Gaussians, corresponding to no rotation. This initialization strategy provides a neutral starting point, allowing subsequent optimization processes to refine the orientations as needed.这部分是相当于是初始化高斯的内容，posed reference depth images $\{D_k\}$,为了方便评估实验，我们使用给定姿态的posed reference depth images来进行初始化gs，然后第三部分应该是优化停止，收敛的描述，，大量实验结果显示大约在100次迭代后总损失基本稳定，并且设置了patience机制，我设置为100次后启动patience机制，如果连续超过patience次数总损失不再下降，就推出优化迭代循环，采用总损失最小值的时候为最佳的估计姿态，这是优化收敛停止的描述，我的描述比较口语化，但你写的必须是符合书面论文要求的，计算机顶会论文的标准，开始你对Localization pipeline三段的英文学术论文写作，前后需要非常的连贯学术化，根据我提供给你所有的资料

### Experements

```GPT
你是ChatGPT，由OpenAI训练的大型语言模型。请仔细遵循用户的指示。使用 Markdown 格式进行回应。用Latex写公式时，公式放在$内返回，确保能用Markdown渲染。请你扮演一名熟知各个研究领域的发展历程和最新进展的高级研究员。

我希望你能担任英语拼写校对和修辞改进的角色。

请严格遵守以下修改要求：

我会发送学术论文的语句或段落给你。请逐句将其中的的词汇和句子替换成更为准确和学术的表达方式，确保意思不变，语言不变，但使其更具学术。

请严格按照下列格式输出回答：

首先给出修改后的全文，语言必须与我发送给你的文本语言相同。
然后使用markdown表格格式逐句输出以下内容：

原文被修改内容，没有被修改的部分则跳过。

修改后的内容，语言必须与我发送给你的文本语言相同。

修改理由，请注意，修改理由必须使用中文输出。

语句通顺，用词准确的部分不进行修改，不在表格里列出。

专业词汇不进行修改，不在表格里列出。

表格中原文整句输出。

示例：

修改后：

<修改后>

解析：

| 原文 | 修改后 | 修改理由 |

|------------------------|-----------------------|---------------------------|

| <原文1> | <修改后1> | <修改理由1> |

| <原文2> | <修改后2> | <修改理由2> |

| <原文3> | <修改后3> | <修改理由3> |

接下来我会发送需要你英语拼写校对和修辞改进的内容，请你开始上述操作。As an experienced academic research writer, your task is to write an "evaluation" part of paper. This work should be detailed, well-researched, and written in an academic style. It needs to provide a comprehensive overview of the subject matter, present a logical argument or analysis, and substantiate it with relevant sources, theories or data. Make sure to incorporate current and relevant references for supporting your points. The language used should be formal, precise, and clear. The document should be formatted according to the applicable academic writing guidelines or style guide. Proofread for clarity, coherence, grammar, and punctuation before submission. here is my chapter:
```

```GPT
As an experienced academic research writer, your task is to write an [introduction/chapter/conclusion] discussing the [topic]. This work should be detailed, well-researched, and written in an academic style. It needs to provide a comprehensive overview of the subject matter, present a logical argument or analysis, and substantiate it with relevant sources, theories or data. Make sure to incorporate current and relevant references for supporting your points. The language used should be formal, precise, and clear. The document should be formatted according to the applicable academic writing guidelines or style guide. Proofread for clarity, coherence, grammar, and punctuation before submission.
```


```GPT
来你要写成像论文那样。一个整段落的去描述前后句子要衔接起来，不要分开分罗列呢，是像人工智能写的，我是让你帮我润色我的论文，你懂我意思吗？按照计算机学术顶会论文的风格来进行写作。
```

```GTP
公式均用$$风格,公式必须用$$包裹而不是/(这种，我要适配obsidian我的笔记
```

```GPT
分析阐述定位方式是什么？定位方式依赖的场景表示是什么？重点解析定位方式的原理和公式，要有分析和对应的公式，公式均用$$风格,公式必须用$$包裹而不是/(这种，我要适配obsidian我的笔记，主要用论文中的公式来解释，你自己也可以加一些数学描述，要指明数学公式是论文中还是你自己写的
```

```GPT
公式均用$$风格，展示关于定位部分的评估标准注意是定位部分，不需要展示重建质量的评估指标，用了哪些及实验数据展示，公式必须用$$包裹而不是/(这种，我要适配obsidian我的笔记,实验数据表格要给全，也要标注是哪个数据集的实验数据表格,每个数据集的表格数据都需要有
```
RMSE

Standard Deviation

![[assets/Pasted image 20240727181353.png|400]]![[assets/Pasted image 20240727181456.png|300]]

*(ATE RMSE [cm] ↓)*
### Replica Dataset

| Methods    | Avg. | R0   | R1   | R2   | Of0  | Of1  | Of2  | Of3  | Of4  |
| ---------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| DROID-SLAM | 0.38 | 0.53 | 0.38 | 0.45 | 0.35 | 0.24 | 0.36 | 0.33 | 0.43 |
| Vox-Fusion | 3.09 | 1.37 | 4.70 | 1.47 | 8.48 | 2.04 | 2.58 | 1.11 | 2.94 |
| NICE-SLAM  | 1.06 | 0.97 | 1.31 | 1.07 | 0.88 | 1.00 | 1.06 | 1.10 | 1.13 |
| ESLAM      | 0.63 | 0.71 | 0.70 | 0.52 | 0.57 | 0.55 | 0.58 | 0.72 | 0.63 |
| Point-SLAM | 0.52 | 0.61 | 0.41 | 0.37 | 0.38 | 0.48 | 0.54 | 0.69 | 0.72 |
| SplaTAM    | 0.36 | 0.31 | 0.40 | 0.29 | 0.47 | 0.27 | 0.29 | 0.32 | 0.55 |

### TUM-RGBD Dataset

| Methods       | Avg.  | fr1/desk | fr1/desk2 | fr1/room | fr2/xyz | fr3/off. |
| ------------- | ----- | -------- | --------- | -------- | ------- | -------- |
| Kintinous     | 4.84  | 3.70     | 7.10      | 7.50     | 2.90    | 3.00     |
| ElasticFusion | 6.91  | 2.53     | 6.83      | 21.49    | 1.17    | 2.52     |
| ORB-SLAM2     | 1.98  | 1.60     | 2.20      | 4.70     | 0.40    | 1.00     |
| NICE-SLAM     | 15.87 | 4.26     | 4.99      | 34.49    | 31.73   | 3.87     |
| Vox-Fusion    | 11.31 | 3.52     | 6.00      | 19.53    | 1.49    | 26.01    |
| Point-SLAM    | 8.92  | 4.34     | 4.54      | 30.92    | 1.31    | 3.48     |
| SplaTAM       | 5.48  | 3.35     | 6.54      | 11.13    | 1.24    | 5.16     |


这是我的ate数据{"columns": ["scenes", "ATEs"], "data": [["freiburg1_desk2", 0.01006467080353563], ["freiburg2_xyz", 0.0024796385908429953], ["freiburg1_desk", 0.009310321468721178], ["freiburg3_long_office_household", 0.011974489286711599], ["freiburg1_room", 0.0066618738092843535]]}

这是我的aae数据{"columns": ["scenes", "AREs"], "data": [["freiburg1_desk2", 1.265491742759908], ["freiburg2_xyz", 0.7890543848912203], ["freiburg1_desk", 1.1263295053841953], ["freiburg3_long_office_household", 0.8082835497537473], ["freiburg1_room", 0.9072185591963973]]}



### reference
References [1] J. Czarnowski, T. Laidlow, R. Clark, and A. J. Davison. Deepfactors: Real-time probabilistic dense monocular SLAM. IEEE Robotics and Automation Letters (RAL), 5(2): 721–728, 2020. [2] Angela Dai, Matthias Nießner, Michael Zollh ̈ ofer, Shahram Izadi, and Christian Theobalt. BundleFusion: Real-time Globally Consistent 3D Reconstruction using On-the-fly Surface Re-integration. ACM Transactions on Graphics (TOG), 36(3):24:1–24:18, 2017. [3] Eric Dexheimer and Andrew J. Davison. Learning a Depth Covariance Function. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [4] J. Engel, V. Koltun, and D. Cremers. Direct sparse odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2017. [5] C. Forster, M. Pizzoli, and D. Scaramuzza. SVO: Fast SemiDirect Monocular Visual Odometry. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2014. [6] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. [7] Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and ShiMin Hu. Di-fusion: Online implicit 3d reconstruction with deep priors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021. [8] M. M. Johari, C. Carta, and F. Fleuret. ESLAM: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [9] M. Keller, D. Lefloch, M. Lambers, S. Izadi, T. Weyrich, and A. Kolb. Real-time 3D Reconstruction in Dynamic Scenes using Point-based Fusion. In Proc. of Joint 3DIM/3DPVT Conference (3DV), 2013. [10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ̈ uhler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023. [11] Leonid Keselman and Martial Hebert. Approximate differentiable rendering with algebraic surfaces. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. [12] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR), 2015. [13] Heng Li, Xiaodong Gu, Weihao Yuan, Luwei Yang, Zilong Dong, and Ping Tan. Dense rgb slam with neural implicit maps. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. [14] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. NeurIPS, 2020. [15] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. 3DV, 2024. [16] J. McCormac, A. Handa, A. J. Davison, and S. Leutenegger. SemanticFusion: Dense 3D semantic mapping with convolutional neural networks. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2017. [17] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. [18] N. J. Mitra, N. Gelfand, H. Pottmann, and L. J. Guibas. Registration of Point Cloud Data from a Geometric Optimization Perspective. In Proceedings of the Symposium on Geometry Processing, 2004. [19] Thomas M ̈ uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG), 2022. [20] R. Mur-Artal and J. D. Tard ́ os. ORB-SLAM2: An OpenSource SLAM System for Monocular, Stereo, and RGB-D Cameras. IEEE Transactions on Robotics (T-RO), 33(5): 1255–1262, 2017. [21] R. Mur-Artal, J. M. M Montiel, and J. D. Tard ́ os. ORBSLAM: a Versatile and Accurate Monocular SLAM System. IEEE Transactions on Robotics (T-RO), 31(5):1147–1163, 2015. [22] R. A. Newcombe. Dense Visual SLAM. PhD thesis, Imperial College London, 2012. [23] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon. KinectFusion: Real-Time Dense Surface Mapping and Tracking. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2011. [24] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. [25] M. Nießner, M. Zollh ̈ofer, S. Izadi, and M. Stamminger. Real-time 3D Reconstruction at Scale using Voxel Hashing. In Proceedings of SIGGRAPH, 2013. [26] Victor Adrian Prisacariu, Olaf K ̈ ahler, Ming-Ming Cheng, Carl Yuheng Ren, Julien P. C. Valentin, Philip H. S. Torr, Ian D. Reid, and David W. Murray. A framework for the volumetric integration of depth images. CoRR, abs/1410.0925, 2014. [27] Erik Sandstr ̈ om, Yue Li, Luc Van Gool, and Martin R. Oswald. Point-slam: Dense neural point cloud-based slam. In Proceedings of the International Conference on Computer Vision (ICCV), 2023. [28] Thomas Sch ̈ ops, Torsten Sattler, and Marc Pollefeys. Surfelmeshing: Online surfel-based mesh reconstruction. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2020. [29] Thomas Sch ̈ops, Torsten Sattler, and Marc Pollefeys. Bad slam: Bundle adjusted direct rgb-d slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. [30] J. Sol a, J. Deray, and D. Atchuthan. A micro Lie theory for state estimation in robotics. arXiv:1812.01537, 2018. [31] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, XiaqingPan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. [32] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A Benchmark for the Evaluation of RGB-D SLAM Systems. In Proceedings of the IEEE/RSJ Conference on Intelligent Robots and Systems (IROS), 2012. [33] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison. iMAP: Implicit mapping and positioning in real-time. In Proceedings of the International Conference on Computer Vision (ICCV), 2021. [34] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. [35] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. Proceedings of the International Conference on Learning Representations (ICLR), 2024. [36] Zachary Teed and Jia Deng. DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras. In Neural Information Processing Systems (NIPS), 2021. [37] Emanuele Vespa, Nikolay Nikolov, Marius Grimm, Luigi Nardi, Paul HJ Kelly, and Stefan Leutenegger. Efficient octree-based volumetric SLAM supporting signed-distance and occupancy mapping. IEEE Robotics and Automation Letters (RAL), 2018. [38] Angtian Wang, Peng Wang, Jian Sun, Adam Kortylewski, and Alan Yuille. Voge: a differentiable volume renderer using gaussian ellipsoids for analysis-by-synthesis. 2022. [39] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [40] T. Whelan, M. Kaess, H. Johannsson, M. F. Fallon, J. J. Leonard, and J. B. McDonald. Real-time large scale dense RGB-D SLAM with volumetric fusion. International Journal of Robotics Research (IJRR), 34(4-5):598–626, 2015. [41] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison. ElasticFusion: Dense SLAM without a pose graph. In Proceedings of Robotics: Science and Systems (RSS), 2015. [42] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. [43] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2022. [44] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. Proceedings of the International Conference on Learning Representations (ICLR), 2024. [45] Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. [46] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. [47] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R Oswald, Andreas Geiger, and Marc Pollefeys. Nicer-slam: Neural implicit scene encoding for rgb slam. International Conference on 3D Vision (3DV), 2024. [48] M. Zwicker, H. Pfister, J. van Baar, and M. Gross. Ewa splatting. IEEE Transactions on Visualization and Computer Graphics, 8(3):223–238, 2002.



## [@engelDirectSparseOdometry2017]
### Frame-to-Frame Pose Estimation and Volumetric Representation in DSO

This document describes a method for visual odometry, Direct Sparse Odometry (DSO), that estimates camera motion and scene geometry from a sequence of images. The following response addresses how DSO achieves frame-to-frame pose estimation and what kind of volumetric representation it relies on for tracking.

**Frame-to-Frame Pose Estimation**

DSO performs frame-to-frame pose estimation using a direct image alignment technique. This involves directly minimizing the photometric error between two frames, without relying on keypoint detection and matching like traditional methods.

Here's how it works:

- **Direct Image Alignment:** For each new frame, DSO performs direct image alignment with a single reference frame, usually the latest keyframe.
- **Semi-Dense Depth Map:** To aid in tracking, the keyframe is accompanied by a semi-dense depth map, created by projecting all active 3D points into the keyframe and slightly dilating them.
- **Multi-Scale Image Pyramid:** Alignment is performed using a multi-scale image pyramid, starting from the coarsest level and refining the estimate at each subsequent level.
- **Constant Motion Model:** An initial guess for the camera pose is obtained using a constant motion model.

**Volumetric Representation for Tracking**

DSO doesn't explicitly rely on a specific volumetric representation like voxels or occupancy grids for tracking. Instead, it leverages a sparse set of 3D points represented as inverse depths relative to a reference frame. These points are selected based on their image gradient magnitude and spatial distribution.

The point cloud obtained through this sparse representation implicitly embodies the scene's volumetric information. This sparse point cloud is used in various ways for tracking:

- **New Frame Tracking:** As described above, projecting the active points into a new keyframe generates a semi-dense depth map, guiding the direct image alignment of subsequent frames.
- **Epipolar Line Search:** New candidate points are tracked in subsequent frames by searching along the epipolar line, constrained by the depth estimate from the previous frame.
- **Outlier Removal:** Depth estimates associated with each 3D point help identify and discard outlier measurements based on their consistency with the estimated geometry.

In summary, while DSO doesn't explicitly use a volumetric representation like voxels, it leverages a sparse 3D point cloud which implicitly contains information about the scene's geometry. This sparse representation, combined with direct image alignment techniques, enables efficient and accurate frame-to-frame pose estimation.


##  [@forsterSVOFastSemidirect2014]
### Frame-to-Frame Pose Estimation in SVO

The paper you provided, "SVO: Fast Semi-Direct Monocular Visual Odometry," describes a method for estimating camera motion using a semi-direct approach that combines aspects of both feature-based and direct methods. Let's break down how SVO achieves frame-to-frame pose estimation.

**SVO does not rely on a volumetric representation for tracking.** Instead, it maintains a sparse map of 3D points reconstructed from the images.

**SVO's frame-to-frame pose estimation involves three main steps:**

1. **Sparse Model-based Image Alignment (Section IV-A):** This step provides an initial estimate of the camera pose relative to the previous frame. It minimises the photometric error between small image patches (4x4 pixels) around feature points in the current frame and their projected locations from the previous frame, given the depth estimate for those points from the map. This process implicitly assumes the epipolar constraint between the frames.
    
    > _"The maximum likelihood estimate of the rigid body transformation Tk,k−1 between two consecutive camera poses minimizes the negative log-likelihood of the intensity residuals"_
    
2. **Relaxation Through Feature Alignment (Section IV-B):** The initial pose estimate is refined by aligning the current frame with the keyframes in the map. For each 3D point in the map visible in the current frame, SVO identifies the keyframe with the closest observation angle to the point. It then performs an affine warp alignment between the image patches in the current frame and the selected keyframe.
    
    > _"To reduce the drift, the camera pose should be aligned with respect to the map, rather than to the previous frame."_
    
3. **Pose and Structure Refinement (Section IV-C):** The final step involves a bundle adjustment (BA) procedure to optimise both the camera pose and the 3D point cloud by minimising the reprojection error. This step can be performed either for the current frame and map or locally for a set of recent keyframes.
    
    > _"In this final step, we again optimize the camera pose Tk,w to minimize the reprojection residuals"_
    

**Impact of Volumetric Representation (Not Applicable to SVO)**

The provided source does not mention the use of a volumetric representation in SVO. Therefore, any discussion about its impact on pose estimation in this context would be purely speculative and outside the scope of the source material.




##  [@czarnowskiDeepfactorsRealtimeProbabilistic2020]

### 1. How is pose optimized? Please explain the pose optimization pipeline in detail.

#### Explanation:
Pose optimization in this paper is achieved through a **factor graph-based optimization** system that incorporates **multiple error factors**. The pipeline begins with each camera frame being represented by a **6 Degrees of Freedom (DoF) pose** \( p_i \) and an associated **depth map code** \( c_i \), which parameterizes the dense geometry. These codes are optimized simultaneously with the poses by minimizing several consistency losses between frames.

The system defines three key factors for optimization:

- **Photometric Factor**: This factor measures the intensity differences between pixels in two frames. Specifically, the intensity of pixels in frame \( j \) is warped to frame \( i \), and the differences between the warped and observed pixel intensities are minimized as part of the optimization process. The error is defined as:  
  > "We minimize various pairwise consistency losses \( e_{ij} \) in order to find the best estimate for scene geometry \( G \) and camera poses" [Czarnowski et al_2020_Deepfactors, page 3](4).

- **Reprojection Factor**: This measures the difference in location of image features detected in multiple frames. Matched image features between frames are reprojected using the relative pose transformation between the frames, and the differences in location are minimized. The reprojection error is defined as:  
  > "We also use the indirect reprojection error widely used in classical structure from motion" [Czarnowski et al_2020_Deepfactors, page 3](4).

- **Geometric Factor**: This factor enforces consistency between depth maps in different frames. The depth map from frame \( j \) is warped into frame \( i \) using the relative pose, and the error between the warped depth map and the original depth map is minimized.  
  > "Another form of consistency can be expressed with differences in scene geometry" [Czarnowski et al_2020_Deepfactors, page 3](4).

All of these factors are integrated into a **factor graph** that represents the relationships between keyframes, codes, and camera poses. The system uses the **iSAM2 algorithm** to efficiently update the factor graph when new observations are introduced:  
> "The mapping step performs batch optimization of all keyframes in the map using standard factor graph software" [Czarnowski et al_2020_Deepfactors, page 5](4).

#### Citations:
- "We minimize various pairwise consistency losses \( e_{ij} \) in order to find the best estimate for scene geometry \( G \) and camera poses" [Czarnowski et al_2020_Deepfactors, page 3](4).
- "We also use the indirect reprojection error widely used in classical structure from motion" [Czarnowski et al_2020_Deepfactors, page 3](4).
- "Another form of consistency can be expressed with differences in scene geometry" [Czarnowski et al_2020_Deepfactors, page 3](4).
- "The mapping step performs batch optimization of all keyframes in the map using standard factor graph software" [Czarnowski et al_2020_Deepfactors, page 5](4).

---

### 2. In this pipeline, are two frames used to estimate the pose of a subsequent frame? If yes, explain how the estimation is performed.

#### Explanation:
Yes, in this system, two or more frames are used to estimate the pose of a subsequent frame. The pipeline uses **pairwise constraints** between frames, which means that the pose of the new frame is estimated by comparing it to the last \( N \) keyframes. The photometric, reprojection, and geometric factors ensure that the system minimizes the errors between the new frame and the previous keyframes, resulting in an optimized pose for the new frame.  
> "Each new keyframe is connected to last \( N \) keyframes in the map using selected pairwise consistency factors" [Czarnowski et al_2020_Deepfactors, page 5](4).

The process of **warping** pixel intensities from previous frames into the new frame’s coordinate system is a key part of this optimization. By minimizing the differences between the warped and observed pixels, the system derives the relative pose between frames:  
> "The relative 6DoF transformation from frame \( i \) to \( j \)" is used to warp pixel coordinates and minimize the photometric and reprojection losses [Czarnowski et al_2020_Deepfactors, page 3](4).

#### Citations:
- "Each new keyframe is connected to last \( N \) keyframes in the map using selected pairwise consistency factors" [Czarnowski et al_2020_Deepfactors, page 5](4).
- "The relative 6DoF transformation from frame \( i \) to \( j \)" is used to warp pixel coordinates and minimize the photometric and reprojection losses [Czarnowski et al_2020_Deepfactors, page 3](4).

---

### 3. Explain the impact of scene representation on pose estimation and optimization. How does scene representation influence the results?

#### Explanation:
The scene representation in this system is based on a **compact, learned code** that encodes the depth maps at each frame. This compact representation greatly impacts pose estimation and optimization by reducing the dimensionality of the optimization problem. Instead of optimizing depth values for each pixel independently, the system optimizes the **latent code** that represents the entire depth map, making the optimization more tractable and efficient:  
> "When optimizing on the code manifold, groups of depth pixels are correlated together which makes the optimization problem more tractable" [Czarnowski et al_2020_Deepfactors, page 3](4).

This representation also helps **improve robustness** in challenging conditions, such as when there is limited photometric information (e.g., in **textureless areas**). The code representation allows the system to enforce depth consistency even in these challenging scenarios:  
> "The geometric error introduces a prior about the world that only a single surface is observed and pins separate depth maps together to form a single reconstruction in textureless areas" [Czarnowski et al_2020_Deepfactors, page 6](4).

By using this compact representation, the system is able to maintain **real-time performance** while still optimizing both the camera poses and the scene geometry:  
> "The optimised GPU usage, efficient implementation and SLAM design choices enable real-time performance" [Czarnowski et al_2020_Deepfactors, page 2](4).

#### Citations:
- "When optimizing on the code manifold, groups of depth pixels are correlated together which makes the optimization problem more tractable" [Czarnowski et al_2020_Deepfactors, page 3](4).
- "The geometric error introduces a prior about the world that only a single surface is observed and pins separate depth maps together to form a single reconstruction in textureless areas" [Czarnowski et al_2020_Deepfactors, page 6](4).
- "The optimised GPU usage, efficient implementation and SLAM design choices enable real-time performance" [Czarnowski et al_2020_Deepfactors, page 2](4).

---

This approach ensures that all key points are directly sourced from the provided paper and that every sentence includes citations as requested.


## [@teedDroidslamDeepVisual2021]
Here are the answers to your questions based on the **DROID-SLAM** paper with citations directly from the text:

---

### 1. **How is pose optimized? Please explain the pose optimization pipeline in detail.**

#### Explanation:
Pose optimization in **DROID-SLAM** is performed using a combination of **recurrent iterative updates** and a **Dense Bundle Adjustment (DBA) layer**. The camera poses and depth maps are iteratively updated through a **ConvGRU-based update operator**, which operates on the co-visibility graph of frames. Each frame in the system is represented by its pose \( G_t \in SE(3) \) and its **inverse depth map**. The update process uses geometric constraints and photometric residuals to jointly optimize both the pose and depth.

- **ConvGRU Update Operator**: The operator performs iterative updates to the current estimates of poses and depths using a combination of **flow features** and **correlation features** between frames.
  > "The update operator is a 3× 3 convolutional GRU with hidden state h" [Teed_Deng_2021_Droid-slam.pdf, page 4](15).
  
- **Dense Bundle Adjustment (DBA) Layer**: This layer optimizes the pose and depth by solving a **Gauss-Newton optimization** over the full frame graph. The cost function incorporates the **revised correspondences** between frames, minimizing the reprojection error for pose and depth updates.
  > "Each update of camera poses and depth maps in DROID-SLAM is produced by a differentiable Dense Bundle Adjustment (DBA) layer" [Teed_Deng_2021_Droid-slam.pdf, page 2](15).

The pipeline operates iteratively until convergence, where the **poses and depths** reach a **fixed point**.
> "Iterative applications of the update operator produce a sequence of poses and depths, with the expectation of converging to a fixed point" [Teed_Deng_2021_Droid-slam.pdf, page 4](15).

#### Citations:
- "The update operator is a 3× 3 convolutional GRU with hidden state h" [Teed_Deng_2021_Droid-slam.pdf, page 4](15).
- "Each update of camera poses and depth maps in DROID-SLAM is produced by a differentiable Dense Bundle Adjustment (DBA) layer" [Teed_Deng_2021_Droid-slam.pdf, page 2](15).
- "Iterative applications of the update operator produce a sequence of poses and depths, with the expectation of converging to a fixed point" [Teed_Deng_2021_Droid-slam.pdf, page 4](15).

---

### 2. **In this pipeline, are two frames used to estimate the pose of a subsequent frame? If yes, explain how the estimation is performed.**

#### Explanation:
Yes, multiple frames are used to estimate the pose of subsequent frames in DROID-SLAM. Specifically, the system builds a **co-visibility frame graph**, where edges are established between frames with **overlapping fields of view**. Each frame is connected to its neighboring frames in the graph, and the **camera pose** is iteratively updated by minimizing the **photometric and geometric residuals** over all edges.

- **Co-visibility Frame Graph**: This graph dynamically connects frames based on the shared points between their fields of view, enabling the system to refine pose estimates through connections to multiple frames.
  > "We adopt a frame-graph (V, E) to represent co-visibility between frames" [Teed_Deng_2021_Droid-slam.pdf, page 3](15).

- **Iterative Updates**: The system uses the **Dense Bundle Adjustment (DBA) layer** to compute pose updates for all frames in the graph. These updates are based on the **correlation volumes** between frames, which capture the visual similarity between neighboring frames.
  > "We iteratively update camera poses and depth... applied to an arbitrary number of frames, enabling joint global refinement of all camera poses and depth maps" [Teed_Deng_2021_Droid-slam.pdf, page 2](15).

#### Citations:
- "We adopt a frame-graph (V, E) to represent co-visibility between frames" [Teed_Deng_2021_Droid-slam.pdf, page 3](15).
- "We iteratively update camera poses and depth... applied to an arbitrary number of frames, enabling joint global refinement of all camera poses and depth maps" [Teed_Deng_2021_Droid-slam.pdf, page 2](15).

---

### 3. **Explain the impact of scene representation on pose estimation and optimization. How does scene representation influence the results?**

#### Explanation:
The **scene representation** in DROID-SLAM is captured through the **inverse depth maps** associated with each frame. This representation significantly influences the accuracy and robustness of pose estimation. By optimizing **pixel-wise depth** and incorporating **optical flow information**, the system can accurately estimate camera poses even in challenging environments.

- **Inverse Depth Representation**: Each frame maintains an **inverse depth map**, which is iteratively updated along with the camera poses. This dense depth representation ensures that depth information is used directly for pose optimization.
  > "For each image t, we maintain two state variables: camera pose Gt ∈ SE(3) and inverse depth dt ∈ RH×W+" [Teed_Deng_2021_Droid-slam.pdf, page 3](15).

- **Impact on Robustness**: The use of a **pixel-wise dense depth map** allows DROID-SLAM to handle challenging visual scenarios, such as **dynamic environments** and **motion blur**, providing robustness and accuracy in various settings.
  > "We optimize over per-pixel depth directly... allowing our network to better generalize to new datasets" [Teed_Deng_2021_Droid-slam.pdf, page 3](15).

- **Geometric Constraints**: By leveraging **geometric constraints** in the DBA layer, DROID-SLAM achieves **global consistency** across frames, reducing errors in long trajectories and improving pose accuracy.
  > "The Dense Bundle Adjustment (DBA) layer... leverages geometric constraints, improves accuracy and robustness" [Teed_Deng_2021_Droid-slam.pdf, page 2](15).

#### Citations:
- "For each image t, we maintain two state variables: camera pose Gt ∈ SE(3) and inverse depth dt ∈ RH×W+" [Teed_Deng_2021_Droid-slam.pdf, page 3](15).
- "We optimize over per-pixel depth directly... allowing our network to better generalize to new datasets" [Teed_Deng_2021_Droid-slam.pdf, page 3](15).
- "The Dense Bundle Adjustment (DBA) layer... leverages geometric constraints, improves accuracy and robustness" [Teed_Deng_2021_Droid-slam.pdf, page 2](15).

---

This detailed breakdown is fully based on the paper and includes citations for every sentence as per your request. Let me know if further clarification is needed!


 ##  [@mccormacSemanticfusionDense3d2017]
### 1. How is pose optimized? Please explain the pose optimization pipeline in detail.

In the **SemanticFusion** pipeline, pose optimization is handled through the **ElasticFusion SLAM** system, which relies on a **combined ICP (Iterative Closest Point) and RGB alignment**. The system tracks the camera pose for each incoming frame \(k\), generating a new pose \(T_{WC}\), where \(W\) refers to the world frame and \(C\) refers to the camera frame【9:16†source】. New surfels (3D points) are added to the map using this updated camera pose, and the existing surfel information is refined using newly obtained data on positions, normals, and colors【9:16†source】. 

Additionally, **ElasticFusion** performs continuous checks for **loop closure events**. When detected, the map is optimized in real-time, immediately applying corrections and deformations to maintain consistency【9:16†source】【9:10†source】. This optimization process is enhanced by **fusing new depth readings**, which adjust the surfels' depth and normal information without disrupting the probability distribution tied to the surfels【9:16†source】.

### 2. In this pipeline, are two frames used to estimate the pose of a subsequent frame? If yes, explain how the estimation is performed.

Yes, in the **SemanticFusion** pipeline, **multiple frames are used** to estimate and refine the pose of subsequent frames. **Pose estimation** is facilitated through **SLAM correspondences**, which allow the system to associate surfels between different frames【9:16†source】【9:5†source】. As each frame provides a new camera pose \(T_{WC}\), the position of every surfel in the map is projected into the new frame via a transformation matrix. This matrix connects the world frame to the camera frame, enabling the fusion of information from multiple frames into the final pose estimate【9:16†source】【9:5†source】.

This method also applies a **Bayesian update** process, which recursively refines the surfel labels and their probability distributions based on frame-wise CNN predictions【9:5†source】.

### 3. Explain the impact of scene representation on pose estimation and optimization. How does scene representation influence the results?

The **scene representation** significantly impacts pose estimation and optimization in **SemanticFusion**. The **surfel-based surface representation** used in ElasticFusion allows for a **deformation** that keeps the representation consistent during pose corrections, particularly when **loop closures** are detected【9:13†source】. This continuous deformation ensures that pose estimation remains accurate even during complex trajectories and significant viewpoint changes【9:13†source】.

Moreover, **scene geometry** provides important regularization for pose estimation, helping to constrain predictions and improve final results. The detailed 3D scene map, created by integrating semantic information from multiple frames, helps reduce uncertainty and refine pose estimates【9:17†source】【9:16†source】.

Finally, **viewpoint variation** plays a critical role in improving pose optimization. The more diverse the viewpoints captured during the trajectory, the better the system can disambiguate poses and enhance overall accuracy【9:17†source】【9:19†source】. This is particularly effective in longer trajectories, where more varied data provides clearer corrections for both pose and scene reconstruction【9:17†source】【9:10†source】.














In contrast to NeRF, 3DGS performs differentiable rasterisation. Similar to regular graphics rasterisations, by iterating over the primitives to be rasterised rather than marching along rays, 3DGS leverages the natural sparsity of a 3D scene and achieves a representation which is expressive to capture high-fidelity 3D scenes while offering significantly faster rendering. Several works have applied 3D Gaussians and differentiable rendering to static scene capture [@keselmanApproximateDifferentiableRendering2022], [@wangVoGEDifferentiableVolume2024], and in particular more recent works utilise 3DGS and demonstrate superior results in vision tasks such as dynamic scene capture [@luitenDynamic3DGaussians2023],[@wu4dGaussianSplatting2024], and 3D generation [[@tangDreamGaussianGenerativeGaussian2024], [@yiGaussianDreamerFastGeneration2024]].
这边可以说点云对齐算法在高斯slam中的应用这些，还有可微分的一些方法

Our method adopts a Map-centric approach, utilising 3D Gaussians as the only SLAM representation. Similar to surfel-based SLAM, we dynamically allocate the 3D Gaussians, enabling us to model an arbitrary spatial distribution in the scene. Unlike other methods such as ElasticFusion [[@whelanElasticFusionDenseSLAM2015]] and PointFusion [@kellerRealtime3dReconstruction2013], however, by using differentiable rasterisation, our SLAM system can capture highfidelity scene details and represent challenging object properties by direct optimisation against information from every pixel.

3DGS performs differentiable rasterisation. Similar to regular graphics rasterisations, by iterating over the primitives to be rasterised rather than marching along rays, 3DGS leverages the natural sparsity of a 3D scene and achieves a representation which is expressive to capture high-fidelity 3D scenes while offering significantly faster rendering. Several works have applied 3D Gaussians and differentiable rendering to static scene capture [11, 38], and in particular more recent works utilise 3DGS and demonstrate superior results in vision tasks such as dynamic scene capture [15, 42, 44] and 3D generation [35, 45]. Our method adopts a Map-centric approach, utilising 3D Gaussians as the only SLAM representation. Similar to surfel-based SLAM, we dynamically allocate the 3D Gaussians, enabling us to model an arbitrary spatial distribution in the scene. Unlike other methods such as ElasticFusion [41] and PointFusion [9], however, by using differentiable rasterisation, our SLAM system can capture highfidelity scene details and represent challenging object properties by direct optimisation against information from every pixe








The classical method for creating a 3D representation was to unproject 2D observations into 3D space and to fuse them via weighted averaging [[@mccormacSemanticfusionDense3d2017], [@newcombeKinectfusionRealtimeDense2011]]. Such an averaging scheme suffers from over-smooth representation and lacks the expressiveness to capture high-quality details. To capture a scene with photorealistic quality, differentiable volumetric rendering [[@niemeyerDifferentiableVolumetricRendering2020]] has recently been popularised with Neural Radiance Fields (NeRF) [[@mildenhallNeRFRepresentingScenes2022]]. Using a single Multi-Layer Perceptron (MLP) as a scene representation, NeRF performs volume rendering by marching along pixel rays, querying the MLP for opacity and colour. Since volume rendering is naturally differentiable, the MLP representation is optimised to minimise the rendering loss using multiview information to achieve high-quality novel view synthesis. The main weakness of NeRF is its training speed. Recent developments have introduced explicit volume structures such as multi-resolution voxel grids [[@fridovich-keilPlenoxelsRadianceFields2022],[@liuNeuralSparseVoxel2020], [@sunDirectVoxelGrid2022]] or hash functions [[@mullerInstantNeuralGraphics2022]] to improve performance. Interestingly, these projects demonstrate that the main contributor to high-quality novel view synthesis is not the neural network but rather differentiable volumetric rendering, and that it is possible to avoid the use of an MLP and yet achieve comparable rendering quality to NeRF [[@fridovich-keilPlenoxelsRadianceFields2022]]. However, even in these systems, per-pixel ray marching remains a significant bottleneck for rendering speed. **This issue is particularly critical in SLAM, where immediate interaction with the map is essential for tracking.** 

The classical method for creating a 3D representation was to unproject 2D observations into 3D space and to fuse them via weighted averaging [16, 23]. Such an averaging scheme suffers from over-smooth representation and lacks the expressiveness to capture high-quality details. To capture a scene with photorealistic quality, differentiable volumetric rendering [24] has recently been popularised with Neural Radiance Fields (NeRF) [17]. Using a single Multi-Layer Perceptron (MLP) as a scene representation, NeRF performs volume rendering by marching along pixel rays, querying the MLP for opacity and colour. Since volume rendering is naturally differentiable, the MLP representation is optimised to minimise the rendering loss using multiview information to achieve high-quality novel view synthesis. The main weakness of NeRF is its training speed. Recent developments have introduced explicit volume structures such as multi-resolution voxel grids [6, 14, 34] or hash functions [19] to improve performance. Interestingly, these projects demonstrate that the main contributor to high-quality novel view synthesis is not the neural network but rather differentiable volumetric rendering, and that it is possible to avoid the use of an MLP and yet achieve comparable rendering quality to NeRF [6]. However, even in these systems, per-pixel ray marching remains a significant bottleneck for rendering speed. This issue is particularly critical in SLAM, where immediate interaction with the map is essential for tracking. 







Dense visual SLAM focuses on reconstructing detailed 3D maps, unlike **sparse SLAM methods which excel in pose estimation** [@engelDirectSparseOdometry2017], [@forsterSVOFastSemidirect2014], [@camposOrbslam3AccurateOpensource2021] but typically yield maps useful mainly for localisation.需要展开介绍传统spare slam中 的姿态对齐和优化方法， In contrast, dense SLAM creates interactive maps beneficial for broader applications, including AR and robotics. Dense SLAM methods are generally divided into two primary categories: Frame-centric and Map-centric. **Frame-centric SLAM minimises photometric error across consecutive frames, jointly estimating per-frame depth and frame-to-frame camera motion.** 不同的侧重点slam方法中的相机定位方法,我们应该是这种方法 **Frame-centric approaches** [[@czarnowskiDeepfactorsRealtimeProbabilistic2020], [@teedDroidslamDeepVisual2021]] are efficient, *as individual frames host local rather than global geometry* (e.g. depth maps), and are attractive for long-session SLAM, but if a dense global map is needed, it must be constructed on demand by assembling all of these parts which are not necessarily fully consistent. In contrast, **Map-centric SLAM uses a unified 3D representation** across the SLAM pipeline, enabling a compact and streamlined system. *Compared to purely local frame-to-frame tracking*, a map-centric approach *leverages global information by tracking against the reconstructed 3D consistent map*. Classical **map-centric** approaches often use voxel grids [[@daiBundleFusionRealTimeGlobally2017], [@newcombeKinectfusionRealtimeDense2011],[@prisacariuFrameworkVolumetricIntegration2014], [@whelanRealtimeLargescaleDense2015]] or points [[@kellerRealtime3dReconstruction2013], [@schopsBadSlamBundle2019], [@whelanElasticFusionDenseSLAM2015]] as the underlying 3D representation. *While voxels enable a fast look-up of features in 3D, the representation is expensive,* and the fixed voxel resolution and distribution are problematic when the spatial characteristics of the environment are not known in advance. On the other hand, a point-based map representation, such as surfel clouds, enables adaptive changes in resolution and spatial distribution by dynamic allocation of point primitives in the 3D space. Such flexibility benefits online applications such as SLAM with deformation-based loop closure [[@schopsBadSlamBundle2019], [@whelanElasticFusionDenseSLAM2015]]. However, optimising the representation to capture high fidelity is challenging due to the lack of correlation among the primitives. Recently, in addition to classical graphic primitives, neural network-based map representations are a promising alternative. iMAP [@sucarImapImplicitMapping2021]demonstrated the interesting properties of neural representation, such as sensible hole filling of unobserved geometry. Many recent approaches combine the classical and neural representations to capture finer details [[@johariEslamEfficientDense2023], [@sandstromPointslamDenseNeural2023], [@zhuNiceslamNeuralImplicit2022], [@zhuNICERSLAMNeuralImplicit2024]]; however, the large amount of computation required for neural rendering makes the live operation of such systems challenging.  



## Data

```json
{
    "Replica": {
        "office0": {
            "PLANE_ICP": {
                "ATE": 0.009218136845309245,
                "AAE": 0.6813786529722292
            },
            "ICP": {
                "ATE": 0.009828920575153843,
                "AAE": 0.6397085400706546
            },
            "HYBRID": {
                "ATE": 0.009260470710823393,
                "AAE": 0.7093803001528893
            },
            "GICP": {
                "ATE": 0.009235954526542398,
                "AAE": 0.7091162762447387
            },
            "ours": {
                "ATE": 0.01136,
                "AAE": 0.0092
            },
        },
        "room1": {
            "PLANE_ICP": {
                "ATE": 0.008545326798500322,
                "AAE": 0.771595331519191
            },
            "HYBRID": {
                "ATE": 0.008305277109664127,
                "AAE": 0.811948281179614
            },
            "GICP": {
                "ATE": 0.008277756846772226,
                "AAE": 0.8120269517624817
            },
            "ICP": {
                "ATE": 0.009345091608532669,
                "AAE": 0.6901135884024677
            },
            "ours": {
                "ATE": 0.01272,
                "AAE": 0.0081
            },
        },
        "office4": {
            "PLANE_ICP": {
                "ATE": 0.013041102162928279,
                "AAE": 0.5583324238016412
            },
            "GICP": {
                "ATE": 0.012844602245591858,
                "AAE": 0.6244782506884774
            },
            "ICP": {
                "ATE": 0.013402050431974146,
                "AAE": 0.4185897042410004
            },
            "HYBRID": {
                "ATE": 0.01288694763489181,
                "AAE": 0.6249887648873795
            },
            "ours": {
                "ATE": 0.01943,
                "AAE": 0.0108
            },
        },
        "office1": {
            "HYBRID": {
                "ATE": 0.005949668611968673,
                "AAE": 0.5412518999531623
            },
            "ICP": {
                "ATE": 0.006257844810962991,
                "AAE": 0.3360501298609102
            },
            "PLANE_ICP": {
                "ATE": 0.005897142489476554,
                "AAE": 0.5215786994390265
            },
            "GICP": {
                "ATE": 0.005905746094078801,
                "AAE": 0.5372479631737159
            },
            "ours": {
                "ATE": 0.00937,
                "AAE": 0.0087
            },
        },
        "room2": {
            "HYBRID": {
                "ATE": 0.011828757316108688,
                "AAE": 0.7811090770352759
            },
            "ICP": {
                "ATE": 0.011167806470795957,
                "AAE": 0.5435629107555225
            },
            "GICP": {
                "ATE": 0.011831545557342829,
                "AAE": 0.7806502271994282
            },
            "PLANE_ICP": {
                "ATE": 0.011864373671664485,
                "AAE": 0.72265162118324
            },
            "ours": {
                "ATE": 0.02052,
                "AAE": 0.0100
            },
        },
        "office2": {
            "ICP": {
                "ATE": 0.011942467747145465,
                "AAE": 0.43359553271679724
            },
            "GICP": {
                "ATE": 0.011754309249873772,
                "AAE": 0.6624031352825694
            },
            "PLANE_ICP": {
                "ATE": 0.011620425231809929,
                "AAE": 0.5820463852187063
            },
            "HYBRID": {
                "ATE": 0.012012526284780035,
                "AAE": 0.667409353130265
            },
            "ours": {
                "ATE": 0.01836,
                "AAE": 0.0107
            },
        },
        "room0": {
            "ICP": {
                "ATE": 0.012862818688285479,
                "AAE": 0.42855613953828664
            },
            "PLANE_ICP": {
                "ATE": 0.012461519643353982,
                "AAE": 0.464926325320993
            },
            "HYBRID": {
                "ATE": 0.012475273309330381,
                "AAE": 0.47551622390311404
            },
            "GICP": {
                "ATE": 0.012502471121917812,
                "AAE": 0.4761471210747908
            },
            "ours": {
                "ATE": 0.01519,
                "AAE": 0.0072
            },
        },
        "office3": {
            "ICP": {
                "ATE": 0.013342029024320344,
                "AAE": 0.2810756854209011
            },
            "PLANE_ICP": {
                "ATE": 0.01425645851000965,
                "AAE": 0.43794648071600284
            },
            "GICP": {
                "ATE": 0.014376723144929757,
                "AAE": 0.44595512322597475
            },
            "HYBRID": {
                "ATE": 0.014990679562373028,
                "AAE": 0.4489020237956806
            },
            "ours": {
                "ATE": 0.02003,
                "AAE": 0.0093
            },
        }
    },
    "TUM": {
        "freiburg1_desk2": {
            "GICP": {
                "ATE": 0.03493094006634798,
                "AAE": 2.09771012942133
            },
            "ours": {
                "ATE": 0.01006467080353563,
                "AAE": 1.265491742759908
            },
            "ICP": {
                "ATE": 0.008263124512777763,
                "AAE": 1.5567344179800904
            },
            "HYBRID": {
                "ATE": 0.01880004539225674,
                "AAE": 1.7910670519987126
            },
            "PLANE_ICP": {
                "ATE": 0.019512051627959458,
                "AAE": 1.618028703652856
            }
        },
        "freiburg1_desk": {
            "ICP": {
                "ATE": 0.007198055538163964,
                "AAE": 1.1812398513177207
            },
            "ours": {
                "ATE": 0.009310321468721178,
                "AAE": 1.1263295053841953
            },
            "PLANE_ICP": {
                "ATE": 0.016589247789088817,
                "AAE": 1.2883301843641115
            },
            "HYBRID": {
                "ATE": 0.018336434170907332,
                "AAE": 1.3878816018601805
            },
            "GICP": {
                "ATE": 0.022647300773953605,
                "AAE": 1.4260948068155348
            }
        },
        "freiburg1_room": {
            "HYBRID": {
                "ATE": 0.013980406196051213,
                "AAE": 1.563696839251442
            },
            "ICP": {
                "ATE": 0.007440724280045329,
                "AAE": 1.3547035114489654
            },
            "PLANE_ICP": {
                "ATE": 0.01607241481816131,
                "AAE": 1.3630038961588138
            },
            "ours": {
                "ATE": 0.0066618738092843535,
                "AAE": 0.9072185591963973
            },
            "GICP": {
                "ATE": 0.027829045732082416,
                "AAE": 1.5940124490160028
            }
        },
        "freiburg2_xyz": {
            "GICP": {
                "ATE": 0.0028694736274749518,
                "AAE": 0.11372516974206438
            },
            "ours": {
                "ATE": 0.0024796385908429953,
                "AAE": 0.7890543848912203
            },
            "ICP": {
                "ATE": 0.0005446733031711187,
                "AAE": 0.138261620929117
            },
            "HYBRID": {
                "ATE": 0.0030505377541225256,
                "AAE": 0.18153819062009052
            },
            "PLANE_ICP": {
                "ATE": 0.0028118848079180655,
                "AAE": 0.1473715536622591
            }
        },
        "freiburg3_long_office_household": {
            "PLANE_ICP": {
                "ATE": 0.008950671671456599,
                "AAE": 0.38056867384114895
            },
            "HYBRID": {
                "ATE": 0.010190616786212098,
                "AAE": 0.5253053983911798
            },
            "ours": {
                "ATE": 0.011974489286711599,
                "AAE": 0.8082835497537473
            },
            "ICP": {
                "ATE": 0.005371213866705301,
                "AAE": 0.3470945447273432
            },
            "GICP": {
                "ATE": 0.009450288874598829,
                "AAE": 0.35480172129819
            }
        }
    }
}
```

::: {.table}
:Replica[@sturmBenchmarkEvaluationRGBD2012] (ATE RMSE ↓\[cm\]).

| Methods   | Avg.    | R0      | R1      | R2      | Of0     | Of1     | Of2     | Of3     | Of4     |
| --------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| ICP       | 1.10186 | 1.28628 | 0.93451 | 1.11678 | 0.98289 | 0.62578 | 1.19425 | 1.33420 | 1.34021 |
| PLANE_ICP | 1.08631 | 1.24615 | 0.85453 | 1.18644 | 0.92181 | 0.58971 | 1.16204 | 1.42565 | 1.30411 |
| HYBRID    | 1.09637 | 1.24753 | 0.83053 | 1.18288 | 0.92605 | 0.59497 | 1.20125 | 1.49907 | 1.28869 |
| GICP      | 1.08411 | 1.25025 | 0.82778 | 1.18315 | 0.92360 | 0.59057 | 1.17543 | 1.43767 | 1.28446 |
| Ours      | 0.01587 | 0.01519 | 0.01272 | 0.02052 | 0.01136 | 0.00937 | 0.01836 | 0.02003 | 0.01943 |
:::

::: {.table}
:TUM[@sturmBenchmarkEvaluationRGBD2012] (ATE RMSE ↓\[cm\]).

| Methods   | Avg.    | fr1/desk | fr1/desk2 | fr1/room | fr2/xyz | fr3/off. |
| --------- | ------- | -------- | --------- | -------- | ------- | -------- |
| ICP       | 0.57636 | 0.71981  | 0.82631   | 0.74407  | 0.05447 | 0.53712  |
| PLANE_ICP | 1.27873 | 1.65892  | 1.95121   | 1.60724  | 0.28119 | 0.89507  |
| HYBRID    | 1.28716 | 1.83364  | 1.88000   | 1.39804  | 0.30505 | 1.01906  |
| GICP      | 1.95454 | 2.26473  | 3.49309   | 2.78290  | 0.28695 | 0.94503  |
| Ours      | 0.80982 | 0.93103  | 1.00647   | 0.66619  | 0.24796 | 1.19745  |
:::

::: {.table}
:Replica[@sturmBenchmarkEvaluationRGBD2012] (AAE RMSE ↓\[cm\]).

| Methods   | Avg.    | R0      | R1      | R2      | Of0     | Of1     | Of2     | Of3     | Of4     |
| --------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| ICP       | 0.47141 | 0.42856 | 0.69011 | 0.54356 | 0.63971 | 0.33605 | 0.43360 | 0.28108 | 0.41859 |
| PLANE_ICP | 0.59256 | 0.46493 | 0.77160 | 0.72265 | 0.68138 | 0.52158 | 0.58205 | 0.43795 | 0.55833 |
| HYBRID    | 0.63256 | 0.47552 | 0.81195 | 0.78111 | 0.70938 | 0.54125 | 0.66741 | 0.44890 | 0.62499 |
| GICP      | 0.63100 | 0.47615 | 0.81203 | 0.78065 | 0.70912 | 0.53725 | 0.66240 | 0.44596 | 0.62448 |
| Ours      | 0.00925 | 0.00720 | 0.00810 | 0.01000 | 0.00920 | 0.00870 | 0.01070 | 0.00930 | 0.01080 |
:::

::: {.table}
:TUM[@sturmBenchmarkEvaluationRGBD2012] (AAE RMSE ↓\[cm\]).

| Methods   | Avg.    | fr1/desk | fr1/desk2 | fr1/room | fr2/xyz | fr3/off. |
| --------- | ------- | -------- | --------- | -------- | ------- | -------- |
| ICP       | 0.91561 | 1.18124  | 1.55673   | 1.35470  | 0.13826 | 0.34709  |
| PLANE_ICP | 0.95946 | 1.28833  | 1.61803   | 1.36300  | 0.14737 | 0.38057  |
| HYBRID    | 1.08990 | 1.38788  | 1.79107   | 1.56370  | 0.18154 | 0.52531  |
| GICP      | 1.11727 | 1.42609  | 2.09771   | 1.59401  | 0.11373 | 0.35480  |
| Ours      | 0.97928 | 1.12633  | 1.26549   | 0.90722  | 0.78905 | 0.80828  |
:::



