#pragma once

/// 从文件读入BAL dataset
class BALProblem {
public:
    /// load bal data from text file
    // explicit：指定构造函数或转换函数 (C++11起)为显式, 即它不能用于隐式转换和复制初始化
    // 可以与常量表达式一同使用. 当该常量表达式为 true 才为显式转换(C++20起)
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);

    ~BALProblem() {//析构函数 delete 指针
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    /// save results to text file
    void WriteToFile(const std::string &filename) const;

    /// save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    void Normalize(); // 数据归一化

    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);//添加高斯噪声

    int camera_block_size() const { return use_quaternions_ ? 10 : 9; } // 返回相机位姿参数的内存大小(是否使用四元数

    int point_block_size() const { return 3; } // 返回观测点的数据内存大小

    int num_cameras() const { return num_cameras_; } // 返回相机数

    int num_points() const { return num_points_; } // 返回路标点数

    int num_observations() const { return num_observations_; } // 返回观测数

    int num_parameters() const { return num_parameters_; } // 返回相机位姿和路标点的个数

    const int *point_index() const { return point_index_; } // 返回观测值（像素坐标）对应的观测路标索引

    const int *camera_index() const { return camera_index_; } // 返回观测值（像素坐标）对应的相机位姿索引

    const double *observations() const { return observations_; } // 存储观测值的首地址

    const double *parameters() const { return parameters_; } // 参数（pose和point）首地址

    const double *cameras() const { return parameters_; } // 相机参数首地址

    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; } // 路标point的首地址

    /// camera参数的起始地址
    // mutable 修饰的代表在使用时会改变原有值
    double *mutable_cameras() { return parameters_; }

    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    double *mutable_camera_for_observation(int i)
    { // 第i个观测值对应的相机参数地址
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double *mutable_point_for_observation(int i)
    { // 第i个观测值对应的路标point
        return mutable_points() + point_index_[i] * point_block_size();
    }

    const double *camera_for_observation(int i) const
    { // 第i个观测值对应的相机参数地址
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation(int i) const
    { // 第i个观测值对应的路标point
        return points() + point_index_[i] * point_block_size();
    }

private:
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    int *point_index_;      // 每个observation对应的point index
    int *camera_index_;     // 每个observation对应的camera index
    double *observations_;
    double *parameters_; // parameters_为txt文件后面的那些行
};
