#pragma once
#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"

namespace myslam {

/**
 * 配置类，使用SetParameterFile确定配置文件
 * 然后用Get得到对应值
 * 单例模式
 */
class Config {
   private:
    static std::shared_ptr<Config> config_;//记录对象被引用的次数，当引用次数为 0 的时候，
    //也就是最后一个指向该对象的共享指针析构的时候，共享指针的析构函数就把指向的内存区域释放掉
    
    cv::FileStorage file_;//cv::FileStorage对象代表一个YAML或XML格式的文件

    Config() {}  // private constructor makes a singleton
   public:
    ~Config();  // close the file when deconstructing

    // set a new config file
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values，读取config文件夹下的数据
    template <typename T>
    static T Get(const std::string &key) {
        return T(Config::config_->file_[key]);
    }
};
}  // namespace myslam

#endif  // MYSLAM_CONFIG_H
