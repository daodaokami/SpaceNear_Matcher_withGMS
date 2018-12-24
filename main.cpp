#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv/cv.hpp>
#include <spatical_subdivision.h>
#include <spacenear_matcher.h>

using namespace std;

int main(int argc, char* argv[]) {
    if(argc < 3) {
        cerr<<"do not have enough params (please input two img path)!"<<endl;
        return -1;
    }
    vector<cv::KeyPoint> kps_1, kps_2;
    cv::Mat desc_1, desc_2;
    string img_path_1 = argv[1];
    string img_path_2 = argv[2];

    cv::Mat img_1 = cv::imread(img_path_1, cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = cv::imread(img_path_2, cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    orb->setFastThreshold(5);
    orb->detectAndCompute(img_1, cv::Mat(), kps_1, desc_1);
    orb->detectAndCompute(img_2, cv::Mat(), kps_2, desc_2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches_all;
    matcher.match(desc_1, desc_2, matches_all);
    cout<<"matches all is "<<matches_all.size()<<endl;
    cout<<img_1.size()<<endl;
    cv::Mat img_kps;
    cv::drawKeypoints(img_1, kps_1, img_kps);
    cv::imshow("img 1",img_1);
    cv::imshow("img_kps_1", img_kps);
    cv::waitKey(0);

    Spatical_Subdivision sp_subdiv(img_1.size(), 80);
    //一项一项的测试
    /**
     * 特征点所在区域划分
     *
     * */
    sp_subdiv.SplitPoints2Index(img_1.size(), kps_1);
    std::vector<std::vector<int>> vIndexes = sp_subdiv.GetvIndexes();
    int count = 0;
    for(int i=0; i<vIndexes.size(); i++){
        cout<<"i "<<i<<" "<<vIndexes[i].size()<<endl;
        if(vIndexes[i].size()!=0)
            count++;
    }
    cout<<"not zero Area "<<count<<endl;
    /**
     *
     * 输入空间的位置,得到邻域
     * ok
     * test true!
     * */
    vector<int> neighbors = sp_subdiv.GetNeighbors(2);
    for(int i=0; i<neighbors.size(); i++){
        cout<<"neighbor i "<<i <<" " <<neighbors[i]<<endl;
    }

    /*cv::KeyPoint kp;
    kp.pt.x = 1; kp.pt.y = 2;
    cv::KeyPoint copy = kp;
    cout<<kp.pt<<" "<<copy.pt<<endl;
    cout<<&kp<<" - "<<&copy<<endl;//地址其实不一样了
    cv::KeyPoint* scopy = &kp;
    cout<<&kp<<" - "<<scopy<<endl;

    cv::Mat m(3,4,CV_32FC1);
    m.at<float>(0, 0) = 0;m.at<float>(0, 1) = 1;m.at<float>(0, 2) = 2;m.at<float>(0, 3) = 3;
    m.at<float>(1, 0) = 4;m.at<float>(1, 1) = 5;m.at<float>(1, 2) = 6;m.at<float>(1, 3) = 7;
    m.at<float>(2, 0) = 8;m.at<float>(2, 1) = 9;m.at<float>(2, 2) = 10;m.at<float>(2, 3) = 11;
    cout<<m<<endl;
    cv::Mat mm = m;//是不一样的空间的?  yes
    m.at<float>(0,0) = -1;
    cout<<&mm<<" "<<&m<<endl;
    cout<<mm<<endl;

    mm.row(0) = m.row(1);
    cout<<mm<<endl;

    cv::Mat mmm(3, 4, CV_32FC1);
    mmm.at<float>(0, 0) = 0;mmm.at<float>(0, 1) = 1;mmm.at<float>(0, 2) = 2;mmm.at<float>(0, 3) = 3;
    mmm.at<float>(1, 0) = 4;mmm.at<float>(1, 1) = 5;mmm.at<float>(1, 2) = 6;mmm.at<float>(1, 3) = 7;
    mmm.at<float>(2, 0) = 8;mmm.at<float>(2, 1) = 9;mmm.at<float>(2, 2) = 10;mmm.at<float>(2, 3) = 11;
    mmm = -mmm;
    cout<<mmm<<endl;

    mmm.row(0).copyTo(m.row(0));
    cout<<mmm<<endl;
    cout<<m<<endl;*/

    vector<int*> vi;
    void func(vector<int*>& vi);
    func(vi);
    cout<<"global "<<*vi[0]<<endl;

    GridNet gridNet, gridNet2;
    sp_subdiv.ComputeGirdAndNeighbors(img_1.size(), kps_1, desc_1, gridNet);
    sp_subdiv.ComputeGirdAndNeighbors(img_2.size(), kps_2, desc_2, gridNet2);
    cout<<"it is over!"<<endl;

    int sum_kps = 0;
    //test show the 9 grid keypoints and the neighbors!!!
    for(int i=0; i<gridNet.gridKeypoints.size(); i++) {
        vector<cv::KeyPoint *> sub_ptrkps = gridNet.gridKeypoints[i];
        vector<cv::KeyPoint> subkps(sub_ptrkps.size());
        //在什么地方重复的添加了特征点?
        for (int j = 0;j < subkps.size(); j++) {
            subkps[j].pt = sub_ptrkps[j]->pt;
        }
        sum_kps += subkps.size();
        cv::Mat subkps_img;
        cv::drawKeypoints(img_1, subkps, subkps_img);
        cv::imshow("sub_kps_img", subkps_img);
        cv::waitKey(30);
    }
    cout<<"all kps "<<kps_1.size()<<" sum_kps "<<sum_kps<<endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    //接下来测试邻域功能
    //首先测试19的邻域,这个邻域比较好,特征点也比较多
    vector<cv::KeyPoint *> sub_ptrkps = gridNet.gridKeypoints[19];
    vector<cv::KeyPoint> subkps(sub_ptrkps.size());
    //在什么地方重复的添加了特征点?
    for (int i = 0;i < subkps.size(); i++) {
        subkps[i].pt = sub_ptrkps[i]->pt;
    }
    cv::Mat img_subkps;
    cv::drawKeypoints(img_1, subkps, img_subkps);
    vector<cv::KeyPoint> sub_neighborkps;
    sub_neighborkps.reserve(300);
    //画出相邻图像中的特征点
    //从当前块的邻域中得到相应的特征点描述子等
    int index = 17;
    cout<<"no "<<index <<" has "<<gridNet. neighbors[index].size()<<" neighbors!"<<endl;
    for(int i=0; i<gridNet.neighbors[index].size(); i++){
        cout<<"grid keypoints size  "<<gridNet.neighbors[index][i]->size()<<endl;
        for(int j=0; j<gridNet.neighbors[index][i]->size(); j++){
            cv::KeyPoint kp;
            kp.pt = (gridNet.neighbors[index][i]->at(j)->pt);
            sub_neighborkps.push_back(kp);
        }
        //size是行*列
        cout<<"desc size "<<gridNet.descs_neighbors[index][i].size<<endl;
    }
    cout<<sub_neighborkps.size()<<endl;
    cv::Mat img_subneib;
    cv::drawKeypoints(img_1, sub_neighborkps, img_subneib);
    cv::imshow("sub", img_subkps);
    cv::imshow("neighbor", img_subneib);
    cv::waitKey(0);

    // 好完成了,像素特征点提取,并且进行了划分,接下来进行数据匹配!!!
    // 进行特征点的匹配!


    return 0;
}

void func(vector<int*>& vi) {
    int a = 1;
    vector<int*> subvi;
    subvi.resize(1);
    subvi[0] = &a;
    vi = subvi;
    cout<<"sub "<<*vi[0]<<endl;
}