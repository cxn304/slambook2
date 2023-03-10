//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

Backend::Backend() {
    /*
    原子操作的数据类型（atomic_bool,atomic_int,atomic_long等等），
    对于这些原子数据类型的共享资源的访问，无需借助mutex等锁机制，
    也能够实现对共享资源的正确访问。
    */
    backend_running_.store(true);// 构造函数中启动优化线程并挂起:通过原子操作实现
    /*
    如果回调函数是一个类的成员函数。
    这时想把成员函数设置给一个回调函数指针往往是不行的
    因为类的成员函数，多了一个隐含的参数this。 
    所以直接赋值给函数指针肯定会引起编译报错。这时候就要用到bind
    bind函数的用法和详细参考：
    https://www.cnblogs.com/jialin0x7c9/p/12219239.html
    thread用法参考：
    https://blog.csdn.net/weixin_44156680/article/details/116260129
    以下代码即是实现了创建一个回调函数为BackendLoop()的线程，并传入类参数
    */
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap() {
    //unique_lock用法参考：
    //https://blog.csdn.net/weixin_44156680/article/details/116260129
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();//notify_one()与notify_all()常用来唤醒阻塞的线程
}

void Backend::Stop() {
    backend_running_.store(false);
    map_update_.notify_one();
    //等待该线程终止，例如，在子线程调用了join（time）方法后，
    //主线程只有等待子线程time时间后才能执行子线程后面的代码。
    backend_thread_.join();
}

void Backend::BackendLoop() {
    while (backend_running_.load()) {//用 load() 函数进行读操作
        std::unique_lock<std::mutex> lock(data_mutex_);
         /*
        std::condition_variable实际上是一个类，是一个和条件相关的类，说白了就是等待一个条件达成。
        wait()用来等一个东西：
        一、有除互斥量以外的第二个参数时：
        如果第二个参数的lambda表达式返回值是false，那么wait()将解锁互斥量，并阻塞到本行
        如果第二个参数的lambda表达式返回值是true，那么wait()直接返回并继续执行。（此时对互斥量上锁！）
        阻塞到其他某个线程调用notify_one()成员函数为止；

        二、无除互斥量以外的第二个参数时：
        如果没有第二个参数，那么效果跟第二个参数lambda表达式返回false效果一样
        wait()将解锁互斥量，并阻塞到本行，阻塞到其他某个线程调用notify_one()成员函数为止。

        重点：显然，阻塞在这里的原因是没有获得互斥量的访问权

        三、当其他线程用notify_one()将本线程wait()唤醒后，这个wait恢复后
        1、wait()不断尝试获取互斥量锁，如果获取不到那么流程就卡在wait()这里等待获取，如果获取到了，那么wait()就继续执行，获取到了锁
        2、如果wait有第二个参数就判断这个lambda表达式。
            a)如果表达式为false，那wait又对互斥量解锁，然后又休眠，等待再次被notify_one()唤醒
            b)如果lambda表达式为true，则wait返回，流程可以继续执行（此时互斥量已被锁住）。
        3、如果wait没有第二个参数，则wait返回，流程走下去（直接锁住互斥量）。

        注意：流程只要走到了wait()下面则互斥量一定被锁住了。

        下面一句实现的功能是：执行到wait语句时直接解锁并堵塞，直到其他线程调用notify_one（），直接锁住互斥量并往下执行
        */
        map_update_.wait(lock);

        /// 后端仅优化激活的Frames和Landmarks
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        Optimize(active_kfs, active_landmarks);
    }
}

void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose 顶点，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->Pose());
        optimizer.addVertex(vertex_pose);
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // 路标顶点，使用路标id索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    // K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();

    // edges
    int index = 1;
    double chi2_th = 5.991;  // robust kernel 阈值
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_) continue;
        unsigned long landmark_id = landmark.second->id_;
        auto observations = landmark.second->GetObs();
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;
            auto feat = obs.lock();
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

            auto frame = feat->frame_.lock();
            EdgeProjection *edge = nullptr;
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }

            // 如果landmark还没有被加入优化，则新加一个顶点
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->Pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            edge->setId(index);
            edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
            edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
            edge->setMeasurement(toVec2(feat->position_.pt));
            edge->setInformation(Mat22::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);
            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);

            index++;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and lanrmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace myslam