#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 * **********************************************/
/*
注意 ：
pose_graph_g2o_SE3.cpp与pose_graph_g2o_lie_algebra.cpp的区别在于
前者使用g2o自带的顶点类型VertexSE3和自带的边类型EdgeSE3
而后者自定义了顶点类型VertexSE3LieAlgebra和边类型EdgeSE3LieAlgebra
两者在误差定义和雅克比计算稍有不同  后者效果更好一点
*/
int main(int argc, char **argv) {
    // if (argc != 2) {
    //     cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
    //     return 1;
    // }
    ifstream fin("./sphere.g2o");
    // if (!fin) {
    //     cout << "file " << argv[1] << " does not exist." << endl;
    //     return 1;
    // }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType; // 块求解器
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())); // 使用L-M方法 定义总求解器
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // SE3 顶点
            g2o::VertexSE3 *v = new g2o::VertexSE3(); // g2o文件的格式见pose_graph_g2o_lie_algebra.cpp  此处不赘述
            int index = 0;   // 使用g2o自带顶点  class无需自己重新实现,每次声明一个index,但是在fin读取时候就替换
            fin >> index;
            v->setId(index);
            v->read(fin); // 在此函数中有顶点初始化
            optimizer.addVertex(v); // 增加顶点
            vertexCnt++;
            if (index == 0) // 固定初始顶点  实际上是提供一个先验信息
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2;//读取了两个顶点后,就要读取后面的
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]); // 关联与该边相连的两个顶点
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin); // 读取观测位姿差 信息矩阵  并初始化边
            optimizer.addEdge(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization(); // 初始化
    optimizer.optimize(30); // 设置迭代次数

    cout << "saving optimization results ..." << endl;
    optimizer.save("result.g2o");

    return 0;
}