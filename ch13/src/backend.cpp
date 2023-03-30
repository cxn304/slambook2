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
        Optimize(active_kfs, active_landmarks);//开始做全局优化
    }
}

void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {
    // setup g2o
    //其实 Backend::Optimize()函数 和前端的 EstimateCurrentPose() 函数流有点类似，
    //不同的地方是，在前端做这个优化的时候，只有一个顶点，也就是仅有化当前帧位姿这一个变量，
    //因此边也都是一元边。在后端优化里面，局部地图中的所有关键帧位姿和地图点都是顶点，
    //边也是二元边，在 g2o_types.h 文件中 class EdgeProjection 的 linearizeOplus()函数中，
    //新增了一项 重投影误差对地图点的雅克比矩阵，187页，公式(7.48)
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;//创建稀疏优化器
    optimizer.setAlgorithm(solver);//打开调试输出

    // pose 顶点，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;//https://www.cnblogs.com/yimeixiaobai1314/p/14375195.html map和unordered_map
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {//遍历关键帧   确定第一个顶点
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->Pose());//keyframe的pose(SE3)是待估计的第一个对象
        optimizer.addVertex(vertex_pose);
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});//插入自定义map类型的vertices 不要make_pair也可以嘛
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
    //std::pair主要的两个成员变量是first和second
    for (auto &landmark : landmarks) {//遍历所有活动路标点,就是最多7个
        if (landmark.second->is_outlier_) continue;//外点不优化,first是id,second包括该点的位置,观测的若干相机的位姿,该点在各相机中的像素坐标
        unsigned long landmark_id = landmark.second->id_;//mappoint的id
        auto observations = landmark.second->GetObs();//得到所有观测到这个路标点的feature，是features
        for (auto &obs : observations) {//遍历所有观测到这个路标点的feature，得到第二个顶点，形成对应的点边关系
            if (obs.lock() == nullptr) continue;//如果对象销毁则继续
            auto feat = obs.lock();
            //weak_ptr提供了expired()与lock()成员函数，前者用于判断weak_ptr指向的对象是否已被销毁，
            //后者返回其所指对象的shared_ptr智能指针(对象销毁时返回”空”shared_ptr)
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

            auto frame = feat->frame_.lock();//得到该feature所在的frame
            EdgeProjection *edge = nullptr;
            if (feat->is_on_left_image_) {//判断这个feature在哪个相机
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }

            // 如果landmark还没有被加入优化，则新加一个顶点
            // 意思是无论mappoint被观测到几次，只与其中一个形成关系
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->Pos());// Position in world，是作为estimate的第二个对象
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);//边缘化
                //简单的说G2O 中对路标点设置边缘化(Point->setMarginalized(true))是为了 在计算求解过程中，
                //先消去路标点变量，实现先求解相机位姿，然后再利用求解出来的相机位姿，反过来计算路标点的过程，
                //目的是为了加速求解，并非真的将路标点给边缘化掉。
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);//增加point顶点
            }

            edge->setId(index);
            edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
            edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
            edge->setMeasurement(toVec2(feat->position_.pt));
            edge->setInformation(Mat22::Identity());//e转置*信息矩阵*e,所以由此可以看出误差向量为n×1,则信息矩阵为n×n
            auto rk = new g2o::RobustKernelHuber();//定义robust kernel函数
            rk->setDelta(chi2_th);//设置阈值
            //设置核函数
            //设置鲁棒核函数，之所以要设置鲁棒核函数是为了平衡误差，不让二范数的误差增加的过快。
            // 鲁棒核函数里要自己设置delta值，
            // 这个delta值是，当误差的绝对值小于等于它的时候，误差函数不变。否则误差函数根据相应的鲁棒核函数发生变化。
            edge->setRobustKernel(rk);
            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);//增加边

            index++;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {//确保内点占1/2以上，否则调整阈值，直到迭代结束
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

    for (auto &ef : edges_and_features) {//根据新的阈值，调整哪些是外点 ，并移除
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

    // Set pose and lanrmark position，这样也就把后端优化的结果反馈给了前端
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());//KeyframesType是unordered_map
    }//unordered_map.at()和unordered_map[]都用于引用给定位置上存在的元素
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());//landmarks:unordered_map<unsigned long, myslam::MapPoint::Ptr>
    }
}

}  // namespace myslam