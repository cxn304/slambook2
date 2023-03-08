//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
    //这是一个已知左右相机的点、已知相机之间位姿关系，求解三维点坐标( x w , y w , z w ) (x_w,y_w,z_w)(x w,y w,z w)的问题
    //points[0]是左相机
    //原理参考https://blog.csdn.net/weixin_43956164/article/details/124266267
    MatXX A(2 * poses.size(), 4);//pose的集合，poses.size()就是2
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();//SE3d::matrix3x4()返回一个3x4的矩阵，表示一个3D的刚体变换，m就是T矩阵去掉最下面一行
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);//Eigen::Block表示一个矩阵或数组的一个矩形部分,这里尺寸为1x4
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();//V矩阵最后一列[X,Y,Z,W]就是方程的解
    //但是SVD的解是一个模为1的特征向量，最终的三维坐标需要将特征向量的前三个除以最后一个
    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量不好，放弃
        return true;
    }
    return false;
}

// converters,将cv的point转成Eigen::Vector2d
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
