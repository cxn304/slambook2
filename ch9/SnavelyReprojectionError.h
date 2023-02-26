#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}
    // 模板:https://blog.csdn.net/qq_52905520/article/details/127455728
    // 有了模板功能，则只需要编写一个函数即可，编译器可以通过输入参数的类型，推断出形参的类型
    template <typename T>
    // bool operator()重写括号函数
    // const的用法:https://blog.csdn.net/weixin_56935264/article/details/125760242
    // 下面是const指针和const数据
    // 代价函数模型中需要重载(仿函数)实现误差项的定义,即在bool operator()中定义误差项
    // 前两个是待预测的值,会给定一个初值
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions); // 得到观测点的像素坐标
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template <typename T>
    // 这个是函数模板,非const指针,const数据
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions)
    {
        // Rodrigues' formula,上面这个是内联函数,防止重复调用带来的栈内存开销
        // 这里面不能有选择语句和循环语句
        T p[3];
        AngleAxisRotatePoint(camera, point, p);// rotation.h里面
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center fo distortion
        T xp = -p[0] / p[2];//归一化坐标平面的x和y
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
        const T &l1 = camera[7];
        const T &l2 = camera[8];
        // 径向畸变
        T r2 = xp * xp + yp * yp;//r的平方,r表示点p与坐标系原点的距离
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);
        // 这部分见式5.12和5.13
        const T &focal = camera[6];
        // 重投影得到像素坐标
        predictions[0] = focal * distortion * xp;//这个数据集的cx和cy为0,这个方程就是投影方程
        predictions[1] = focal * distortion * yp;

        return true;
    }
    // 返回一个可自动求导的Ceres代价函数
    // AutoDiffCostFunction 2:dimension of residual, 9:dimention of x, 3:dimension of y
    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        /*
            创建一个自动求导的匿名对象
            自动求导的模板参数：误差类型（即代价函数模型）,输入维度,输出维度（有几个输出写几个）
            参数 ： 代价函数模型的匿名对象
        */
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

