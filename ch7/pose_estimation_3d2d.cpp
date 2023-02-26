#include <iostream>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>
#include "opencv2/imgcodecs/legacy/constants_c.h"


using namespace std;
using namespace cv;

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
// vector<Eigen::Vector2d>;的定义在下面
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
// aligned_allocator:标准的定义容器方法，只是我们一般情况下定义容器的元素都是C++中的类型，
// 所以可以省略，这是因为在C++11标准中，aligned_allocator管理C++中的各种
// 数据类型的内存方法是一样的，可以不需要着重写出来。但是在Eigen管理内存和C++11
// 中的方法是不一样的，所以需要单独强调元素的内存分配和管理
void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

int main(int argc, char **argv) {
  // if (argc != 5) {
  //   cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
  //   return 1;
  // }
  //-- 读取图像
  Mat img_1 = imread("1.png", CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread("2.png", CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点以及cv::Mat的6种创建方法
  // https://blog.csdn.net/u012058778/article/details/90764430
  Mat d1 = imread("1_depth.png", CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 数据自定义矩阵Mat创建
  vector<Point3f> pts_3d; // cv::Point3f
  vector<Point2f> pts_2d; // 对于用户来说最重要的是得到点在基准坐标系下的坐标pts_3d和其他相机的2d像素坐标
  for (DMatch m:matches) {
    // 使用cv::Mat::ptr进行高效的像素访问,第一幅图的行和列对应y和x
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)   // bad depth
      continue;
    float dd = d / 5000.0;//这里可能是米为单位
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//先转到归一化平面上
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);//将第二幅相机中keypoint的像素坐标加入
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV的PnP求解可选择EPNP，DLS等方法
  Mat R;
  cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;

  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  for (size_t i = 0; i < pts_3d.size(); ++i) {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }
  // 上面的循环说明VecVector3d就是Eigen::Vector3d,但是在typedef时候不一样
  cout << "calling bundle adjustment by gauss newton" << endl;
  Sophus::SE3d pose_gn;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

  cout << "calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
  return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;//自己定义的Eigen变量
  typedef Eigen::Matrix<double,6,6> Vextor66;
  const int iterations = 10;
  double cost = 0, lastCost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; iter++) {
    Vextor66 H = Vextor66::Zero();
    // 初始化拟hessian矩阵
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // compute cost,points_3d储存的是相机1的三维坐标,pc则是变换后的
    for (int i = 0; i < points_3d.size(); i++) {
      Eigen::Vector3d pc = pose * points_3d[i]; // pose是T矩阵,李代数的SE3
      double inv_z = 1.0 / pc[2];// pc就是书上的p一撇
      double inv_z2 = inv_z * inv_z; // 1/z和1/z2
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
      //上式是投影的u和v
      Eigen::Vector2d e = points_2d[i] - proj;//观测值减去投影值(预测值)

      cost += e.squaredNorm();//每个点的cost求和
      Eigen::Matrix<double, 2, 6> J; // 式7.46
      J << -fx * inv_z,0,fx * pc[0] * inv_z2,fx * pc[0] * pc[1] * inv_z2,-fx - fx * pc[0] * pc[0] * inv_z2,fx * pc[1] * inv_z,
          0,-fy * inv_z,fy * pc[1] * inv_z2,fy + fy * pc[1] * pc[1] * inv_z2,-fy * pc[0] * pc[1] * inv_z2,-fy * pc[0] * inv_z; // 式7.46
      // H为6x6
      H += J.transpose() * J;//每个点计算得到的H要求和,因为是基于所有点的批次下降
      b += -J.transpose() * e;// 6x2 * 2x1 = 6x1
    }

    Vector6d dx;
    dx = H.ldlt().solve(b);//式6.32求delta x

    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // update your estimation p88的56行
    pose = Sophus::SE3d::exp(dx) * pose;// 李代数(向量)通过指数映射到(李群)式4.25
    lastCost = cost;

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
    if (dx.norm() < 1e-6) {
      // converge
      break;
    }
  }

  cout << "pose by g-n: \n" << pose.matrix() << endl;//进行pose的优化
}

/// vertex and edges used in g2o ba,继承了BaseVertex这个类
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // 内存对齐宏，重载了new函数，使得分配的对象都是16Bytes对齐的
  virtual void setToOriginImpl() override
  { // 使用override显式地声明虚函数覆盖
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3，这里是更新预测T(SE3)值
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override { return false; }

  virtual bool write(ostream &out) const override { return false; }
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 实际上这个宏定义就是用来重构new的，
  //也就是自己实现了一个内存对齐的malloc和free，而c++17之后加入了标准，也就不需要在添加这个宏定义了。
  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }

  virtual bool read(istream &in) override {return false;}

  virtual bool write(ostream &out) const override { return false; }

private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {

  // 构建图优化，先设定g2o,需要优化的是pose,是6维,输入是3维
  // 1,2:创建blocksolver并使用linearsolver对其进行初始化
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  // 并使用BlockSolver的unique_ptr指针初始化，
  // 而BlockSolver又是使用LinearSolver的unique_ptr来初始化的
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;   // 图模型创建图优化的核心:稀疏优化器(SparseOptimizer）
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // vertex 定义图的顶点和边，并添加到SparseOptimizer中
  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose,new一个顶点对象并指向它
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  // K  .at是opencv mat用于索引的方法
  Eigen::Matrix3d K_eigen;
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges 往图中增加边，设置边的属性，然后加入到优化器中
  int index = 1;
  for (size_t i = 0; i < points_2d.size(); ++i) {
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);//3d点往2d的投影
    edge->setId(index);
    edge->setVertex(0, vertex_pose); // p191,设置连接的顶点,这里只有一个顶点,所以是第0个顶点,vertex_pose是定义的Vertex对象的指针
    edge->setMeasurement(p2d);       // 观测数值
    edge->setInformation(Eigen::Matrix2d::Identity()); // 信息矩阵:协方差矩阵之逆（对角阵的逆是对角阵的各个元素的倒数组成的矩阵）
    optimizer.addEdge(edge);
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true); // 打开调试输出
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
  pose = vertex_pose->estimate();
}
