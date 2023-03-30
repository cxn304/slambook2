/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/map.h"
#include "myslam/feature.h"

namespace myslam {
//把当前帧插入到局部地图中,同时检测关键帧数量是否大于7,大于7了就移除一个关键帧
void Map::InsertKeyFrame(Frame::Ptr frame) {
    current_frame_ = frame;
    if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {// 如果当前的keyframe存在且是最后一个的话
        keyframes_.insert(make_pair(frame->keyframe_id_, frame));//KeyframesType是unordered_map类型,可以直接调用make_pair生成pair对象
        active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));//insert()方法可以将pair类型的键值对元素添加到unordered_map容器中
    } else {
        keyframes_[frame->keyframe_id_] = frame;//否则将这个frame设置成keyframes_的当前id
        active_keyframes_[frame->keyframe_id_] = frame;
    }

    if (static_cast<int>(active_keyframes_.size()) > num_active_keyframes_) {
        RemoveOldKeyframe();//如果当前激活的关键帧大于 num_active_keyframes_ (默认是7)，就删除掉老的或太近的关键帧
    }
}
// 在三角化的过程中插入地图点,因为三角化可以求得3d地图点坐标
void Map::InsertMapPoint(MapPoint::Ptr map_point) {
    if (landmarks_.find(map_point->id_) == landmarks_.end()) {//这里比较的是当前mappoint是否是landmarks_中最后一个mappoint
        landmarks_.insert(make_pair(map_point->id_, map_point));
        active_landmarks_.insert(make_pair(map_point->id_, map_point));
    } else {
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }//对于insert方法，如果map中不存在key，则采用拷贝构造函数创建val临时对象（make_pair过程），再采用拷贝构造函数创建map中的val对象。
    //对于[]方法，如果map中不存在key，则采用默认构造函数创建map中val对象，再采用赋值运算符赋值；如果原来存在key，则直接采用赋值运算符赋值。
}

void Map::RemoveOldKeyframe() {
    if (current_frame_ == nullptr) return;
    // 寻找与当前帧最近与最远的两个关键帧
    double max_dis = 0, min_dis = 9999;
    double max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame_->Pose().inverse();
    for (auto& kf : active_keyframes_) {
        if (kf.second == current_frame_) continue;
        auto dis = (kf.second->Pose() * Twc).log().norm();
        if (dis > max_dis) {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        if (dis < min_dis) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    const double min_dis_th = 0.2;  // 最近阈值
    Frame::Ptr frame_to_remove = nullptr;
    if (min_dis < min_dis_th) {
        // 如果存在很近的帧，优先删掉最近的
        frame_to_remove = keyframes_.at(min_kf_id);
    } else {
        // 否则就删掉最远的，这里的思想是，帧很近的话，加入进来优化没有意义，不是很近的话，就将重合度与其他帧相比较小的帧删除
        frame_to_remove = keyframes_.at(max_kf_id);
    }

    LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
    // remove keyframe and landmark observation
    active_keyframes_.erase(frame_to_remove->keyframe_id_);
    for (auto feat : frame_to_remove->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }
    for (auto feat : frame_to_remove->features_right_) {
        if (feat == nullptr) continue;
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }

    CleanMap();
}

// 删除了关键帧后,必然有一些landmark不见了,所以要删除
void Map::CleanMap() {
    int cnt_landmark_removed = 0;
    for (auto iter = active_landmarks_.begin();
         iter != active_landmarks_.end();) {// active_landmarks_是unordered_map,他里面有迭代器
        if (iter->second->observed_times_ == 0) {
            iter = active_landmarks_.erase(iter);// 删除了关键帧后,如果某些active_landmark没有被各个帧的相机观测到,就要删除
            cnt_landmark_removed++;
        } else {
            ++iter;//这里的iter不是一个数字,相当于unordered_map中的元素
        }
    }
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}

}  // namespace myslam
