//
// Created by lut on 18-12-25.
//

#ifndef SPATIALSUBDIVISION_KEYPOINTS_MAP_H
#define SPATIALSUBDIVISION_KEYPOINTS_MAP_H

#include <unordered_map>
#include <vector>
#include <opencv2/core/types.hpp>
using namespace std;

struct Index_Kp{
    int index;
    cv::KeyPoint* ptr_kp;
};
class KeyPoints_Map{
private:
    typedef int Index;
    unordered_map<float, vector<Index_Kp>> unm_v_ikps;  //有一定的可能会遇见重复的元素,所以在唯一位置的时候,用一个vector 存储这个index和对应的keypoint
                                                        //大多数的大小都是1,很难出现都一样的情况
                                                        //这个容器本身的大小,一开始就可以确定的
    //需要通过keypoints的数据,进行快速的查询与
public:
    float compress_func_1(float x, float y){
        return x+y+x*y;
    }//重复的概率应该是比较小的
    void CreateUnorder_Map(vector<cv::KeyPoint>& kps);
};

#endif //SPATIALSUBDIVISION_KEYPOINTS_MAP_H
