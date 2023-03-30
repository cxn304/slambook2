//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"

namespace myslam {
//定义只需要指明文件路径
VisualOdometry::VisualOdometry(std::string &config_path)
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    // read from config file
    //判断一下这个config_file_path_是否存在，
    //同时将配置文件赋给Config类中的cv::FileStorage file_,便于对文件操作
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }
    //1.模板函数自动类型推导调用Config::Get("dataset_dir")；
    //2.模板函数具体类型显示调用Config::Get<std::string>("dataset_dir")；
    dataset_ =
        Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    CHECK_EQ(dataset_->Init(), true);//通过Dataset::Init()函数设置相机的内参数，以及双目相机的外参数

    // create components and links
    //接下来按照逻辑关系一层层的确立联系，
    //一个完整的VO包含前端,后端,地图,可视化器等模块，因此有下述创建代码
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));
    //后端类的定义中用到了相机类和地图类，
    //所以要将后端类与相机类和地图类连接起来
    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));//左相机和右相机
    //对于可视化器来说，只要有地图就可以，它只是将地图可视化，所以不需要其它模块，只需将其与地图模块连接在一起
    viewer_->SetMap(map_);

    return true;
}

void VisualOdometry::Run() {
    while (1) {
        LOG(INFO) << "VO is running";
        //这里的主过程执行在这条if语句中,
        //每次做条件判断都需要执行Step()，即步进操作，如果步进出问题，则跳出死循环while (1)
        if (Step() == false) {
            break;
        }
    }

    backend_->Stop();
    viewer_->Close();

    LOG(INFO) << "VO exit";
}
//前端的主要函数、主逻辑就是Frontend::AddFrame()
bool VisualOdometry::Step() {
    // Step就是要对帧做处理,读进来新的帧等
    Frame::Ptr new_frame = dataset_->NextFrame();//从数据集中读出下一帧
    if (new_frame == nullptr) return false;//这个数据集跑完了，没有下一帧了

    auto t1 = std::chrono::steady_clock::now();//计时
    bool success = frontend_->AddFrame(new_frame);//将新的一帧加入到前端中，进行跟踪处理,帧间位姿估计
    auto t2 = std::chrono::steady_clock::now();//前端的主要函数、主逻辑就是Frontend::AddFrame()
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    return success;
}

}  // namespace myslam
