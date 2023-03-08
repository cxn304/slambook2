//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
    /*
    最大特征点数量 num_features， 
    角点可以接受的最小特征值 检测到的角点的质量等级，角点特征值小于qualityLevel*最大特征值的点将被舍弃 0.01 
    角点最小距离 20  
    */
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}
//在增加某一帧时，根据目前的状况选择不同的处理函数
bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;
    //Track()是Frontend的成员函数,status_是Frontend的数据,可以直接使用
    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}
//在执行Track之前，需要明白，Track究竟在做一件什么事情
//Track是当前帧和上一帧之间进行的匹配
//而初始化是某一帧左右目（双目）之间进行的匹配
bool Frontend::Track() {
    //先看last_frame_是不是正常存在的
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());//给current_frame_当前帧的位姿设置一个初值
    }

    int num_track_last = TrackLastFrame();//使用光流法得到前后两帧之间匹配特征点并返回匹配数
    tracking_inliers_ = EstimateCurrentPose();//接下来根据跟踪到的内点的匹配数目，可以分类进行后续操作，估计当前帧的位姿

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();
    DetectFeatures();  // detect new features

    // track in right image
    FindFeaturesInRight();
    // triangulate map points
    TriangulateNewPoints();
    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}
//在InsertKeyFrame函数中出现了一个三角化步骤，
//这是因为当一个新的关键帧到来后，我们势必需要补充一系列新的特征点，
// 此时则需要像建立初始地图一样，对这些新加入的特征点进行三角化，求其3D位置
int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;//三角化成功的点的数目
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        //遍历左目的特征点
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                //注意这里与初始化地图不同 triangulation计算出来的点pworld，
                //实际上是相机坐标系下的点，所以需要乘以一个TWC
                //但是初始化地图时，是第一帧，一般以第一帧为世界坐标系
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);//设置mapoint类中的坐标
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);//增加mappoint类中的对应的那个feature（左右目）
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

// g2o做优化,加入顶点和边进行优化
int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();//Camera类的成员函数K()

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;//features 存储的是左相机的特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();//weak_ptr是有lock()函数的
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);//只有一个顶点,第一个数是0
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));//测量值是图像上的点
            //图中的Q就是信息矩阵，为了表示我们对误差各分量重视程度的不一样。 
            // 一般情况下，我们都设置这个矩阵为单位矩阵，表示我们对所有的误差分量的重视程度都一样。
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);//鲁棒核函数
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;//重投影误差边界值，大于这个就设置为outline
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        //总共优化了40遍，以10遍为一个优化周期，对outlier进行一次判断
        //舍弃掉outlier的边，随后再进行下一个10步优化
        vertex_pose->setEstimate(current_frame_->Pose());//这里的顶点是SE3位姿,待优化的变量
        optimizer.initializeOptimization();
        optimizer.optimize(10);// 每次循环迭代10次
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {// 特征点本身就是异常点，计算重投影误差
                e->computeError();
            }
            // （信息矩阵对应的范数）误差超过阈值，判定为异常点，并计数，否则恢复为正常点
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                // 设置等级  一般情况下g2o只处理level = 0的边，设置等级为1，下次循环g2o不再优化异常值
                //这里每个边都有一个level的概念，
                //默认情况下，g2o只处理level=0的边，在orbslam中，
                // 如果确定某个边的重投影误差过大，则把level设置为1，
                //也就是舍弃这个边对于整个优化的影响
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };
            //后20次不设置鲁棒核函数了，意味着此时不太可能出现大的异常点
            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());//保存优化后的位姿

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();
    //清除异常点 但是只在feature中清除了
    //mappoint中仍然存在，仍然有使用的可能
    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

//该函数的实现其实非常像FindFeaturesInRight(),
//不同的是一个在左右目之间找，另一个在前后帧之间找
int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    //遍历上一帧中的所有左目特征
    for (auto &kp : last_frame_->features_left_) {
        //判断该特征有没有构建出相应的地图点
        //对于左目图像来说，我们可以将其用于估计相机pose，但是不一定左目图像中的每一个点都有mappoint
        //MapPoint的形成是需要左目和同一帧的右目中构成对应关系才可以，有些左目中的feature在右目中没有配对，就没有Mappoint
        //但是没有Mappoint却不代表这个点是一个outlier
        if (kp->map_point_.lock()) {
            // use project point
            //对于建立了Mappoint的特征点而言 
            //kps_current的初值是通过world2pixel转换得到的
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            //使用new动态创建结构体变量时，必须是结构体指针类型。访问时，普通结构体变量使用使用成员变量访问符"."，
            //指针类型的结构体变量使用的成员变量访问符为"->"。
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            //对于没有建立Mappoint的特征点而言 
            //kps_current的初值是kps_last
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            //status[i]=true则说明跟踪成功有对应点，false则跟踪失败没找到对应点
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    //int num_features_left = DetectFeatures();
    //一个frame其实就是一个时间点，
    //里面同时含有左，右目的图像，以及对应的feature的vector
    //这一步在提取左目特征，通常在左目当中提取特征时特征点数量是一定能保证的。
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}
//检测当前帧的做图的特征点，并放入feature的vector容器中
int Frontend::DetectFeatures() {
    //掩膜，灰度图，同时可以看出，DetectFeatures是对左目图像的操作
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        //在已有的特征附近一个矩形区域内将掩膜值设为0
        //即在这个矩形区域中不提取特征了，保持均匀性，并避免重复
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    //detect函数，第三个参数是用来指定特征点选取区域的，一个和原图像同尺寸的掩膜，其中非0区域代表detect函数感兴趣的提取区域，
    //相当于为detect函数明确了提取的大致位置
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

//找到左目图像的feature之后，就在右目里面找特征点
int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        //遍历左目特征的特征点（feature）
        kps_left.push_back(kp->position_.pt);//feature类中的keypoint对应的point2f
        auto mp = kp->map_point_.lock();//feature类中的mappoint
        if (mp) {
            // use projected points as initial guess
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left image
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;//光流跟踪成功与否的状态向量（无符号字符），成功则为1,否则为0
    Mat error;
    //进行光流跟踪，从这条opencv光流跟踪语句我们就可以知道，
    //前面遍历左目特征关键点是为了给光流跟踪提供一个右目初始值
    //OPTFLOW_USE_INITIAL_FLOW使用初始估计，存储在nextPts中;
    //如果未设置标志，则将prevPts复制到nextPts并将其视为初始估计。
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    //右目中光流跟踪成功的点
    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            //KeyPoint构造函数中7代表着关键点直径
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            //指明是右侧相机feature
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            //左右目匹配失败
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

//现在左目图像的特征提取出来了，并根据左目图像的特征对右目图像做了特征的光流跟踪，
//找到了对应值，当对应数目满足阈值条件时，我们可以开始建立
//初始地图
bool Frontend::BuildInitMap() {
    //构造一个存储SE3的vector，里面初始化就放两个pose，一个左目pose，一个右目pose，
    //看到这里应该记得，对Frame也有一个pose，Frame里面的
    //pose描述了固定坐标系（世界坐标系）和某一帧间的位姿变化
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;//初始化的路标数目
    //遍历左目的feature
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;//右目没有对应点，之前设置左右目的特征点数量是一样的
        // create map point from triangulation
        //对于左右目配对成功的点，三角化它
        //points中保存了双目像素坐标转换到相机（归一化）坐标
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};//这里的depth默认为1.0
        //待计算的世界坐标系下的点
        Vec3 pworld = Vec3::Zero();
        //每一个同名点都进行一次triangulation
        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam