**注：**
+ **1.** 本仓库 forked from [shichaoy/cube_slam](https://github.com/shichaoy/cube_slam)    
Paper: **CubeSLAM: Monocular 3D Object SLAM**, IEEE Transactions on Robotics 2019, S. Yang, S. Scherer  [**PDF**](https://arxiv.org/abs/1806.00557)

+ **2.** 本代码于 18 年底注释，对应作者 18 年 6 月份提交的代码，不包含作者 19 年新增的代码

+ **3.** 本人对 cube slam 代码和相关论文的[总结笔记](https://wym.netlify.app/categories/cube-slam/)

+ **4.** Readme 中 3.1, 3.2, 3.3 的图是我将 cubeslam 移植到 ORB-SLAM2 中的效果，不包含在本仓库的注释代码中

+ **5.** 本人受 Cube SLAM 启发的工作，欢迎批评指正     
Wu Y, Zhang Y, Zhu D, et al. **EAO-SLAM: Monocular Semi-Dense Object SLAM Based on Ensemble Data Association**[J]. arXiv preprint arXiv:2004.12730, 2020. [[**PDF**](https://arxiv.org/abs/2004.12730)] [[**Code**](https://github.com/yanmin-wu/EAO-SLAM)] [[**YouTube**](https://youtu.be/pvwdQoV1KBI)] [[**bilibili**](https://www.bilibili.com/video/av94805216)]. [Submited to IROS 2020]

## 0. object_slam

[^_^]:![](https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/result/result_compare.png?raw=true)

<img src="https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/result/result_compare.png?raw=true" width="600">

## 1. line_lbd
[^_^]:![](https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/result/line/LSD_edline%20.png?raw=true)

<img src="https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/result/line/LSD_edline%20.png?raw=true" width="600">

## 2. single image
[^_^]:![](https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/single%20view.png?raw=true)

<img src="https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/single%20view.png?raw=true" width="600">

## 3. combine with ORB-SLAM2
### 3.1 tum dataset

[![](https://media.giphy.com/media/dxaMwcKw9sriiT4xIl/giphy.gif)](https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/190524tum.gif)

<img src="https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/190525tum.gif?raw=true" width="600">

### 3.2 kitti dataset

[![](https://media.giphy.com/media/KEGOPljrGWFqJY5MJK/giphy.gif)](https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/191007all_out.gif)

### 3.3 real world

[![](https://media.giphy.com/media/fX37X2fpeihIgw2qyZ/giphy.gif)](https://github.com/wuxiaolang/Cube_SLAM_wu/blob/master/wu/190716realworld.gif)

---
wuyanminmax@gmail.com
