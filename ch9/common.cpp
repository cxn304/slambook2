#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.h"
#include "rotation.h"
#include "random.h"
/*
Map定义
Map(PointerArgType dataPtr, Index rows, Index cols, 
const StrideType& stride = StrideType())
可以看出,构建map变量,需要三个信息:指向数据的指针,构造矩阵的行数和列数
map相当于引用普通的c++数组,进行矩阵操作,而不用copy数据
*/
// 一般来说类的属性成员都应设置为private, public只留给那些被外界用来调用的函数接口, 
// 但这并非是强制规定, 可以根据需要进行调整

typedef Eigen::Map<Eigen::VectorXd> VectorRef;//不定的vector长度
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef; // 表示VectorXd的这个向量是const的

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value); // 将文件中format为*format的数据读出来存在value里面
    if (num_scanned != 1)
        std::cerr << "Invalid UW data file. ";
}

void PerturbPoint3(const double sigma, double *point) {
    for (int i = 0; i < 3; ++i) //添加方差为sigma的高斯噪声
        point[i] += RandNormal() * sigma;
}

double Median(std::vector<double> *data) {
    int n = data->size();     // 由begin返回的迭代器指向第一个元素+n/2
    std::vector<double>::iterator mid_point = data->begin() + n / 2; // 返回向量头指针，指向第一个元素，并加上n/2表示中位数的指针
    std::nth_element(data->begin(), mid_point, data->end()); // nth_element会排序,这个就是取中间的点
    return *mid_point;                                       // nth_element默认的升序排序作为排序规则
}

/************************************BAL数据集说明***********************************************************/
/*
我们使用了针孔相机模型，我们为每个相机估计了一些参数，旋转矩阵R，平移矩阵t,焦距f,径向失真参数K1,K2,将3D点投影到相机中
的公式为：
P  =  R * X + t       (conversion from world to camera coordinates)//把世界坐标转换为相机坐标
p  = -P / P.z         (perspective division)//相机坐标归一化处理
u' =  f * r(p) * p    (conversion to pixel coordinates)//转换得到像素坐标 u=fx*X/Z+cx
r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
这给出了像素投影，其中图像的原点是图像的中心，正x轴指向右，正y轴指向上（此外，在相机坐标系中，正z- 轴向后指向，
因此相机正在向下看负z轴，如在OpenGL中那样。
数据格式：
<num_cameras> <num_points> <num_observations>
<camera_index_1> <point_index_1> <x_1> <y_1>
...
<camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
<camera_1>
...
<camera_num_cameras>
<point_1>
...
<point_num_points>
其中，相机和点索引从0开始。每个相机是一组9个参数 - R，t，f，k1和k2。 旋转R被指定为罗德里格斯的向量。
例：
1. BAL数据集说明
第一行：
16 22106 83718
16个相机，22106个点，共进行83718次相机对点的观测
第2行到83719行：
6 18595     3.775000e+01 4.703003e+01
第6个相机观测18595个点，得到的相机的观测数据为3.775000e+01 4.703003e+01
第83720行到83720+16*9=83864
共16个相机的9纬参数：-R（3维），t（3维），f（焦距），k1,k2畸变参数
第83864到83864+3*22106=150182
22106个点的三维坐标
*/
BALProblem::BALProblem(const std::string &filename, bool use_quaternions) {
    FILE *fptr = fopen(filename.c_str(), "r");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    };

    // This wil die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_cameras_); // 都是private的数据,已经定义到common.h中了
    FscanfOrDie(fptr, "%d", &num_points_);  // 读取总的路标数
    FscanfOrDie(fptr, "%d", &num_observations_); // 读取总的观测数据个数

    std::cout << "Header: " << num_cameras_
              << " " << num_points_
              << " " << num_observations_;
    // num_observations_这些是从txt文件的第一行读出来的
    point_index_ = new int[num_observations_]; // 这是新建数组的操作//取出3D路标点的标号
    camera_index_ = new int[num_observations_]; // 相机的标号
    observations_ = new double[2 * num_observations_]; // 观测的像素点

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_; // 每个相机9个参数，每个路标3个参数
    parameters_ = new double[num_parameters_];            // 参数的总大小

    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i); // camera_index_是一个数组,camera_index_是数组头
        FscanfOrDie(fptr, "%d", point_index_ + i);  // 第几个路标
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j); // 观测到的像素坐标
        }
    }
    // 每个相机是一组9个参数，-R:3维(罗德里格斯向量)  t:3维  f,k1,k2。后面是3D路标的数据3维
    for (int i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }

    fclose(fptr);

    use_quaternions_ = use_quaternions;
    if (use_quaternions) {
        // Switch the angle-axis rotations to quaternions.
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
        double *quaternion_parameters = new double[num_parameters_];
        double *original_cursor = parameters_;
        double *quaternion_cursor = quaternion_parameters;
        for (int i = 0; i < num_cameras_; ++i) {
            AngleAxisToQuaternion(original_cursor, quaternion_cursor); // 将轴角转换为四元数
            quaternion_cursor += 4;
            original_cursor += 3;
            for (int j = 4; j < 10; ++j)
            { // 其他参数 t（3维），f（焦距），k1,k2畸变参数 直接赋值
                *quaternion_cursor++ = *original_cursor++;
            }
        }
        // Copy the rest of the points.
        for (int i = 0; i < 3 * num_points_; ++i)
        { // 三维点坐标直接赋值
            *quaternion_cursor++ = *original_cursor++;
        }
        // Swap in the quaternion parameters.
        delete[]parameters_;
        parameters_ = quaternion_parameters;
    }
}

// 输出到普通文件
void BALProblem::WriteToFile(const std::string &filename) const {
    FILE *fptr = fopen(filename.c_str(), "w"); // 打开可写文件

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);
    // 每行输出那个相机观测那个路标点 的 观测值 x,y（四个数据）
    for (int i = 0; i < num_observations_; ++i) {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j) {
            fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }
    // 输出相机坐标到文件
    for (int i = 0; i < num_cameras(); ++i) {
        double angleaxis[9];
        if (use_quaternions_) {
            //OutPut in angle-axis format.
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis); // 统一轴角方式保存
            // 这里用memcpy()函数进行复制，从 parameters_ + 10 * i + 4 位置开始的 6 * sizeof(double)
            // 内存空间的数据放入起始地址为angleaxis + 3的内存空间里
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
        } else {
            // 这里用memcpy()函数进行复制，从 parameters_ + 9 * i位置开始的 9 * sizeof(double) 
            // 内存空间的数据放入起始地址为angleaxis 的内存空间里
            memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
        }
        for (int j = 0; j < 9; ++j) {
            fprintf(fptr, "%.16g\n", angleaxis[j]); // 输出到文件
        }
    }
    // 输出观测路标点到文件
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }

    fclose(fptr); // 打开文件就要关闭文件
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare
void BALProblem::WriteToPLYFile(const std::string &filename) const {
    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    // 创建两个数组，用于承接CameraToAngelAxisAndCenter()解析出来的相机旋转姿态和相机位置中心
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras(); ++i)
    { // 循环写入，首先写入的相机中心点参数，个数控制肯定是相机数据个数
        const double *camera = cameras() + camera_block_size() * i; // cameras都是在common.h定义好的,直接找内存地址的
        // 从cameras头指针开始，每次步进相机维度，这里为9，就读到了每个相机参数
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // 用CameraToAngelAxisAndCenter()函数将从相机参数中解析出来相机姿势和相机位置。当然这里只用位置了。
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n'; // 坐标依次写入文件，再加上颜色数据，最后来个回车
    }

    // Export the structure (i.e. 3D Points) as white points.
    // 相机写完是路标点，用路标个数控制循环次数
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i)
    { // 同样，从路标数据开头位置处，依次偏移路标维度
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            of << point[j] << ' ';
        }
        of << " 255 255 255\n"; // 加上颜色，最后要有回车
    }
    of.close();
}
// camera数据中的旋转向量以及平移向量解析相机世界坐标系的姿态,(依旧是旋转向量)和位置（世界坐标系下的XYZ）
// 具体参数说明：
// camera要解析的相机参数，前三维旋转，接着三维平移
// angle_axis解析出的相机姿态承接数组，也是旋转向量形式
// center是相机原点在世界坐标系下的定义
void BALProblem::CameraToAngelAxisAndCenter(const double *camera,
                                            double *angle_axis,
                                            double *center) const {
    VectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        QuaternionToAngleAxis(camera, angle_axis);
    } else {
        angle_axis_ref = ConstVectorRef(camera, 3);
    }
    // c = -R't
    // 如何计算center
    // center是相机原点在世界坐标系下的定义
    // PW_center:世界坐标系下的相机坐标
    // PC_center:相机坐标系下的相机原点坐标（0,0,0）
    // 根据相机坐标系与世界坐标系的转换关系：PW_center×R+t=PC_center
    // PW_center= -R't
    // 旋转向量的反向过程（求逆）和旋转向量取负一样。
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint(inverse_rotation.data(),
                         camera + camera_block_size() - 6,
                         center);
    VectorRef(center, 3) *= -1.0;
}

void BALProblem::AngleAxisAndCenterToCamera(const double *angle_axis,
                                            const double *center,
                                            double *camera) const {
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        AngleAxisToQuaternion(angle_axis, camera);
    } else {
        VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * c
    AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

void BALProblem::Normalize() {
    // Compute the marginal median of the geometry
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    // mutable_points():获取路标3D点的位置  即parameters_ 中首个3d坐标的地址
    double *points = mutable_points(); // double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i];// points是所有观测点的总数
        }
        median(i) = Median(&tmp);// i表示x,y,z,求他们的中值,即所有点云中最中间的点
    }

    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        tmp[i]=(point-median).lpNorm<1>();//求以中点为中心的归一化坐标，1范数表示向量中所有元素绝对值之和
    }
    // 关于.lpNorm<1>()参见下面网址帖子，是一个范数模板函数
    // http://blog.csdn.net/skybirdhua1989/article/details/17584797
    // 这里用的是L1范数:||x||为x向量各个元素绝对值之和。
    // 简单数一下p范数：向量各元素绝对值的p阶和的p阶根
    // lpNorm<>()函数定义是这么说的：returns the p-th root of the sum of the p-th powers of the absolute values
    // 很明显，p为1的话就是各元素绝对值之和，为2就是模长

    const double median_absolute_deviation = Median(&tmp);

    // Scale so that the median absolute deviation of the resulting
    // reconstruction is 100

    const double scale = 100.0 / median_absolute_deviation;

    // X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    double *cameras = mutable_cameras(); //{ return parameters_; }
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i)
    {                                                       // camera中是9个连续的参数
        double *camera = cameras + camera_block_size() * i; // camera_block_size:{ return use_quaternions_ ? 10 : 9; }
        CameraToAngelAxisAndCenter(camera, angle_axis, center);//计算第i个camera的center
        // center = scale * (center - median), 
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median); 
        // 3是the size of the vector expression
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);
    // 扰动
    double *points = mutable_points();
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            PerturbPoint3(point_sigma, points + 3 * i);//给观测路标增加噪声
        }
    }

    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        // Perturb in the rotation of the camera in the angle-axis
        // representation
        // 轴角[angle-axis]
        // 这里相机是被分成两块, 旋转和平移旋转是考到四元数形式, 増加了ー步用
        // camera Toangelaxisandcenter()从camera中取出三维的angle_axis,
        // 然后派加噪声, 派加完后再用 AngleAxisAndCenterToCamera()重构camera:参数 
        // 平移部分就直接用 Perturbpoint3() 添加了
        if (rotation_sigma > 0.0) {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        if (translation_sigma > 0.0)
            PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
}
