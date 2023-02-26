#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include "opencv2/imgcodecs/legacy/constants_c.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  // if (argc != 3) {
  //   cout << "usage: feature_extraction img1 img2" << endl;
  //   return 1;
  // }
  //-- 读取图像
  Mat img_1 = imread("./1.png", CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread("./2.png", CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  //-- 初始化
  std::vector<KeyPoint> keypoints_1, keypoints_2;//cv::KeyPoint
  Mat descriptors_1, descriptors_2;//cv::Mat
  Ptr<FeatureDetector> detector = ORB::create(); // cv::Ptr类就看成一个cv的
  // 一个智能指针,在适当的时间能自动删除指向的对象
  Ptr<DescriptorExtractor> descriptor = ORB::create();//cv::ORB
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  // cv::DescriptorMatcher
  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); // chrono是c++中的时间库,提供计时,时钟等功能
  detector->detect(img_1, keypoints_1); // keypoints在这一步已经计算出来
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据第一步计算的角点位置计算BRIEF描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  Mat outimg1; // cv::drawKeypoints
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg1);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;//cv::DMatch
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches); // matches.size()刚好500个
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离 std::minmax_element
  // 先是min,再max,所以,返回的类中的数据成员first对应着最小值,second对应着最大值
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);//如果之后某个变量没有用到,c++会释放掉
  //描述子间距离越小,表示匹配越好
  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- 第五步:绘制匹配结果
  // https://blog.csdn.net/u011028345/article/details/73185166
  // Opencv中Mat中元素的值读取方法总结
  // https://blog.csdn.net/weixin_40331125/article/details/107387479
  Mat img_match;//cv::Mat就是可以用来画图的矩阵了
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  imshow("all matches", img_match);
  imshow("good matches", img_goodmatch);
  waitKey(0);

  return 0;
}
