#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用李代数表达位姿图，节点和边的方式为自定义
 * **********************************************/

typedef Matrix<double, 6, 6> Matrix6d;

// 给定误差求J_R^{-1}的近似,&是取地址符
Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    // 此处详细推导见P272 J的近似可以选择两种  此处都有实现 但选择了近似为I（误差接近于0）
    // matrix.block(i,j,p,q),p,q表示block的大小,i,j表示从矩阵中的第几个元素开始向右向下开始算起
    // SO3.log()就可以将SO3上的元素对数表示到so3中，其中so3的定义依然是Vector3d,其实就是Eigen中的类
    // SO3d::hat()可以实现so3->SO3的转换
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
    // J = J * 0.5 + Matrix6d::Identity();
    J = Matrix6d::Identity(); // 在定义该矩阵变量时，创建一个同尺寸同数据类型的单位阵，对其初始化
    return J;
}

// 李代数顶点
typedef Matrix<double, 6, 1> Vector6d;

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 重写读函数
    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];//将7个数据读到data里面
        // 注意SE3的格式:旋转在前（四元数实部在前 虚部在后）平移在后
        setEstimate(SE3d(
            Quaterniond(data[6], data[3], data[4], data[5]),
            Vector3d(data[0], data[1], data[2])
        ));
        return true;
    }

    virtual bool write(ostream &os) const override {
        os << id() << " "; // 位姿编号
        Quaterniond q = _estimate.unit_quaternion(); // 四元数,格式需要转变
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
        return true;
    }

    virtual void setToOriginImpl() override {
        _estimate = SE3d(); // 设置初始值
    }

    // 左乘更新
    virtual void oplusImpl(const double *update) override {
        Vector6d upd;
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3d::exp(upd) * _estimate; // 扰动模型,将李代数转为李群再左乘估计值,就能更新估计值
    }
};

// 两个李代数节点之边
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize(); // 归一化
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2]))); // 设置测量值
         /*
        信息矩阵 代表不确定性
        原文链接 ：https://blog.csdn.net/fly1ng_duck/article/details/101236559?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522160518749619195264726280%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=160518749619195264726280&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-101236559.first_rank_ecpm_v3_pc_rank_v2&utm_term=g2o%E4%BF%A1%E6%81%AF%E7%9F%A9%E9%98%B5&spm=1018.2118.3001.4449
        信息矩阵是协方差矩阵的一个逆矩阵
        信息矩阵在计算条件概率分布明显比协方差矩阵要方便，显然，协方差矩阵要求逆矩阵，所以时间复杂度是O(n^3). 
        之后我们可以在图优化slam中可以看到，
        因为图优化优化后的解是无穷多个的，比如说x1->x2->x3, 每个xi相隔1m这是我们实际观测出来的，
        优化后，我们会得出永远得不出x1 x2 x3的唯一解，因为他们有可能123 可能是234 blabla
        但是如果我们提供固定值比如说x2 坐标是3那么解那么就有唯一解234，提供固定值x2这件事情其实就是个先验信息，
        提供先验信息，求分布，那就是条件分布，也就是这里我们要用到信息矩阵。
        为什么我们需要information matrix 去表征这个uncertainty呢？
        原因就是我们的系统可能有很多传感器，比如说有两个传感器，一个传感器很精确，另外一个很垃圾。
        那么精确那个对应的information matrix里面的系数可能是很大，
        记住！！，这里是越大越好，因为它是协方差矩阵的逆矩阵。
        信息矩阵转置后不变，为什么呢？因为这是通常假定我们的传感器之间是独立的。所以中间两项可以合在一起。

        在stream流类型中，有一个成员函数good().用来判断当前流的状态（读写正常（即符合读取和写入的类型)，没有文件末尾）
        对于类读写文件fstream ifstream ofstream以及读写字符串流stringstream istringstream ostringstream等类型。
        都用good()成员函数来判断当前流是否正常。

        //从sphere.g2o文件中可知
        // 信息矩阵=[ 10000 0      0     0     0     0
        //           0     10000 0     0     0      0
        //           0     0     10000 0     0      0
        //           0     0     0     40000 0      0
        //           0     0     0     0     40000  0
        //           0     0     0     0     0    40000] 6*6
        */
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    virtual bool write(ostream &os) const override {
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *> (_vertices[0]);
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *> (_vertices[1]);
        os << v1->id() << " " << v2->id() << " "; // 与该边相连的两个位姿编号
        SE3d m = _measurement;                    // 边的测量值
        Eigen::Quaterniond q = m.unit_quaternion(); // 四元数   （eigen中正常四元数存储 实部+虚部）
        os << m.translation().transpose() << " ";   // 平移
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix
        for (int i = 0; i < information().rows(); i++) // 信息矩阵只存储右上阵
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    // 误差计算与书中推导一致，_measurement就是q和t，Tij表示Ti到Tj间的运动,用四元数q和平移t来表示
    // 误差项_error=Tij.inverse() * Ti.inverse() * Tj
    virtual void computeError() override {
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();//Ti
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();//Tj
        //事实上,Tij=Ti.inverse()*Tj,但是测量的Tij和右边是不相等的,用他的逆乘以右边就是误差
        _error = (_measurement.inverse() * v1.inverse() * v2).log();              //_measurement:Tij
    }                                 // SO3.log()就可以将SO3上的元素对数表示到so3中

    // 雅可比计算
    virtual void linearizeOplus() override {
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3d::exp(_error));
        // 尝试把J近似为I？
        // 需要计算两个雅克比矩阵（Ti和Tj）  只相差一个负号
        // 这里可以看出v2是Tj
        _jacobianOplusXi = -J * v2.inverse().Adj(); //p272式10.9,可以看出需要先求J
        _jacobianOplusXj = J * v2.inverse().Adj();  // p272式10.10
    }
};

int main(int argc, char **argv) {
    // if (argc != 2) {
    //     cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
    //     return 1;
    // }
    ifstream fin("sphere.g2o");
    // if (!fin) {
    //     cout << "file " << argv[1] << " does not exist." << endl;
    //     return 1;
    // }

    // 设定g2o
    /*
    在十四讲中
    1、拟合曲线的时候，是这样设置的：typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;
    2、之后，当利用BA来将相机的9维参数和3维路标点作为优化变量的时候（这里顶点是相机参数和路标点位置，边是像素坐标），
       又变成了：typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3> > BalBlockSolver。
    3、而在位姿图（pose graph）里面，顶点是位姿，边也是位姿，
       变成了typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,6>> Block。

    先说结论，P,L代表你定义的边所对应的两个顶点的维度，block里面只包含顶点维度，至于边的维度是隐含在边的定义里面的。
    所以你看2里面，一条边两头分别是cam(9维度)和point(3维)，这条边本身只有2维，代表像素坐标，
    error[0,1] = camproject(cam, point)[0,1] - observation[0,1].
    再看3，边的两头分别是SE3(平移加旋转6维)，所以block<6,6>.
    最后1的特殊性在于它是一条unary edge，所以只有一个头，顶点是abc[3]，
    所以error = abc[0] *x^2 + abc[1] * x + abc[2] - observation.   所以一般设置block的时候第二个就是1了
    */
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType; // 块求解器
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())); 
        // 设置总求解器（块求解器（线性求解器））  求解方式L-M
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    /*        
    .g2o文件的数据格式
    1.顶点    ---相机坐标
    VERTEX_SE3:QUAT
    点的索引值        平移                         旋转
    point id      t.x     t.y    t.z        qx     qy    qz     qw
    2.边
    EDGE_SE3:QUAT
    idFrom  idTo    t.x     t.y    t.z      qx    qy    qz     qw           信息矩阵的右上角
    */
    vector<VertexSE3LieAlgebra *> vectices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // 顶点
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin); // 在自定义顶点类中VertexSE3LieAlgebra重写read函数并调用setEstimate实现顶点的初始化
            optimizer.addVertex(v); // optimizer会将所有顶点加进去
            vertexCnt++;
            vectices.push_back(v); // vectices是vector,类型是VertexSE3LieAlgebra
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]); // setVertex是BaseBinaryEdge内置函数
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;

    // 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现
    // 伪装成 SE3 顶点和边，让 g2o_viewer 可以认出
    ofstream fout("result_lie.g2o"); // 输出文件流
    for (VertexSE3LieAlgebra *v:vectices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e:edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}
