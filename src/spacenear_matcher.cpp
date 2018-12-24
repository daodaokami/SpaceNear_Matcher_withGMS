//
// Created by lut on 18-12-22.
//

#include <spacenear_matcher.h>

void merge_kps_descs(std::vector<std::vector<cv::KeyPoint*>*> *ptr_v_ptr_vkps, std::vector<cv::Mat>& descs,
                     std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs);

void merge_kps_descs(std::vector<cv::KeyPoint*> *ptr_vkps, cv::Mat& descs, std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs);

SpaceNear_Matcher::SpaceNear_Matcher(GridNet *gridNet_1, GridNet *gridNet_2) {
    this->gridNet_1 = gridNet_1;
    this->gridNet_2 = gridNet_2;
}

void SpaceNear_Matcher::SpaceNearMatcher(std::vector<cv::DMatch> &all_matches) {
    // SpaceNear_Matcher
    int grid_size = this->gridNet_1->gridKeypoints.size();

}

void SpaceNear_Matcher::SpaceNearMatcher(int index, std::vector<cv::DMatch> &sub_matches) {
    std::vector<cv::KeyPoint*> *ptr_vkps_1 = &this->gridNet_1->gridKeypoints[index];//获取了第index块的当前特征点
    cv::Mat ptr_descs_1 = this->gridNet_1->gridDescs[index];//本身就是浅拷贝

    //找img2 中的相邻特征点与相邻的描述子
    std::vector<std::vector<cv::KeyPoint*>* > v_ptr_vkps_2;
    int neighbor_size = this->gridNet_2->neighbors[index].size();
    std::vector<cv::Mat> v_descs_2;
    v_ptr_vkps_2.resize(neighbor_size);
    v_descs_2.resize(neighbor_size);
    for(int neighbor = 0; neighbor<neighbor_size; neighbor++){
        v_ptr_vkps_2[neighbor] = this->gridNet_2->neighbors[index][neighbor];
        v_descs_2[neighbor] = this->gridNet_2->descs_neighbors[index][neighbor];
    }
    //把这些分散的数据整合成一串的数据
    std::vector<cv::KeyPoint> kps_1, kps_2;
    cv::Mat desc_1, desc_2;

    merge_kps_descs(ptr_vkps_1, ptr_descs_1, kps_1, desc_1);
    merge_kps_descs(&v_ptr_vkps_2, v_descs_2, kps_2, desc_2);
    
}

void merge_kps_descs(std::vector<cv::KeyPoint*> *ptr_vkps, cv::Mat& descs, std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs){
    cdescs = descs;
    kps.resize(ptr_vkps->size());
    for(int i=0; i<ptr_vkps->size(); i++){
        cv::KeyPoint kp;
        kp.pt = (*ptr_vkps)[i]->pt;
        kp.octave = (*ptr_vkps)[i]->octave;
        kp.angle = (*ptr_vkps)[i]->angle;
        kp.response = (*ptr_vkps)[i]->response;
        kps[i] = kp;
    }
    //完成融合与构造了
}

void merge_kps_descs(std::vector<std::vector<cv::KeyPoint*>*> *ptr_v_ptr_vkps, std::vector<cv::Mat>& descs,
                        std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs){
    int sum_kps = 0;
    for(int i=0; i<ptr_v_ptr_vkps->size(); i++){
        sum_kps += (*ptr_v_ptr_vkps)[i]->size();
    }
    kps.resize(sum_kps);
    for(int i=0; i<ptr_v_ptr_vkps->size(); i++){
        int count = 0;
        //即存在多少的邻居!!!
        for(int j=0; j<(*ptr_v_ptr_vkps)[i]->size(); j++){
            cv::KeyPoint kp;
            kp.pt = (*(*ptr_v_ptr_vkps)[i])[j].pt;
            kp.octave = (*(*ptr_v_ptr_vkps)[i])[j].octave;
            kp.angle = (*(*ptr_v_ptr_vkps)[i])[j].angle;
            kp.response = (*(*ptr_v_ptr_vkps)[i])[j].response;
            kps[count] = kp;
            count ++;
        }
        cdescs.push_back(descs[i]);
    }
}
