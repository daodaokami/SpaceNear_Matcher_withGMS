//
// Created by lut on 18-12-25.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "datas_map.h"
using namespace std;

int main(int argc, char* argv[]){
    float fm = 2.50, fz = 4.92;
    float f1 = fz/fm;
    float f2 = 1.968;
    if(f1 == f2){
        cout<<"true"<<endl;
    }
    cout<<f1<<endl;

    if(argc < 2){
        cerr<<"params input the image path to the file!"<<endl;
        return -1;
    }
    string path = argv[1];
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    vector<cv::KeyPoint> kps;
    cv::Mat desc;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    orb->setFastThreshold(5);
    orb->detectAndCompute(img, cv::Mat(), kps, desc);
    cout<<"kps.size "<<kps.size()<<endl;

    cv::Mat img_kps;
    cv::drawKeypoints(img, kps, img_kps);
    cv::imshow("img_kps", img_kps);
    cv::waitKey(0);

    int preSize = kps.size();
    KeyPoints_Map kps_map(preSize);
    kps_map.CreateUnorder_Map(kps);

    //如何遍历一个unordermap的所有元素
    for(unordered_map<int, vector<Index_Kp>>::iterator iter=kps_map.GetMap()->begin();
            iter!=kps_map.GetMap()->end(); iter++){
        int index = iter->first;
        cout<<"-------------- "<<index<<" --------------"<<endl;
        vector<Index_Kp>* ptr_vIkps = &iter->second;
        for(int i=0; i<ptr_vIkps->size(); i++){
            cout<<"the point actually "<<(*ptr_vIkps)[i].index<<" point position "<<(*ptr_vIkps)[i].ptr_kp->pt<<endl;
        }
        cout<<"!!!!!!!!!!!!!                !!!!!!!!!!!!"<<endl;
    }
    //这里就简历某一特征点的索引

    return 0;
}