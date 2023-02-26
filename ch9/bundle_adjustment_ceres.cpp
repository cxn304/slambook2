#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    // if (argc != 2) {
    //     cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
    //     return 1;
    // }

    //BALProblem bal_problem(argv[1]);
    BALProblem bal_problem("problem-16-22106-pre.txt"); // 这一步将数据读取到bal_problem中
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size(); // point数据占几个int
    const int camera_block_size = bal_problem.camera_block_size(); // pose数据占几个int
    double *points = bal_problem.mutable_points();                 // 存储路标点point的首地址
    double *cameras = bal_problem.mutable_cameras();               // 存储相机位姿pose的首地址

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations(); // 存储观测值的首地址
    ceres::Problem problem;                                  // 构建ceres的最小二乘问题

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function; // 代价函数
        // 这个循环主要目的是AddResidualBlock
        //  Each Residual block takes a point and a camera as input
        //  and outputs a 2 dimensional Residual,observations_x,observations_y
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
        /*
        在代价函数的计算模型中实现  用于添加误差项时的第一个参数 实现自动求导
        */
        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0); // 鲁棒核函数

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        // 得到与观测值匹配的相机位姿和观测点3d点
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        problem.AddResidualBlock(cost_function, loss_function, camera, point); // 添加误差项
        // 参数 ：： 代价函数的对象  ， 核函数（不使用时nullptr） ，待估计的参数（camera, point）（可多个）
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // 采用schur消元求解增量方程
    options.minimizer_progress_to_stdout = true;                        // 输出到cout
    ceres::Solver::Summary summary;                                     // 优化信息
    ceres::Solve(options, &problem, &summary);                // 开始优化（参数：配置器 ，最小二乘问题。优化器）
    std::cout << summary.FullReport() << "\n";                // 输出结果
}