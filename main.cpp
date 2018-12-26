#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv/cv.hpp>
#include <spatical_subdivision.h>
#include <spacenear_matcher.h>
#include <chrono>
#include <datas_map.h>
#include "gms_matcher.h"

using namespace std;

//改进点,运动趋势来,进行删除
void GmsMatch(Mat &img1, Mat &img2, vector<cv::KeyPoint>& kps1, vector<cv::KeyPoint>& kps2,
              vector<cv::DMatch>& all_matches);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

void runImagePair(cv::Mat& img1, cv::Mat& img2, vector<cv::KeyPoint>& kps1, vector<cv::KeyPoint>& kps2,
    vector<cv::DMatch>& all_matches) {
    //time cost Most in the detector and match
    //GMS just cost 1~2 ms,it is so fast and good to do with slam
    //how it performance in the uniform features
    GmsMatch(img1, img2, kps1, kps2, all_matches);
}

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

    Spatical_Subdivision sp_subdiv(img_1.size(), 120); 
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
    cout<<"img1 not zero Area "<<count<<endl;

    Spatical_Subdivision sp_subdiv2(img_2.size(), 80);
    sp_subdiv2.SplitPoints2Index(img_2.size(), kps_2);
    std::vector<std::vector<int>> vIndexes2 = sp_subdiv2.GetvIndexes();
    int count2 = 0;
    for(int i=0; i<vIndexes2.size(); i++){
        cout<<"i "<<i<<" "<<vIndexes2[i].size()<<endl;
        if(vIndexes[i].size()!=0)
            count2++;
    }
    cout<<"img2 not zero Area "<<count2<<endl;
    /**
     *
     * 输入空间的位置,得到邻域
     * ok
     * test true!
     * */
    /*vector<int> neighbors = sp_subdiv.GetNeighbors(2);
    for(int i=0; i<neighbors.size(); i++){
        cout<<"neighbor i "<<i <<" " <<neighbors[i]<<endl;
    }*/

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

    /*vector<int*> vi;
    void func(vector<int*>& vi);
    func(vi);
    cout<<"global "<<*vi[0]<<endl;
*/
    GridNet gridNet, gridNet2;
    sp_subdiv.ComputeGirdAndNeighbors(img_1.size(), kps_1, desc_1, gridNet);
    sp_subdiv.ComputeGirdAndNeighbors(img_2.size(), kps_2, desc_2, gridNet2);
    cout<<"it is over!"<<endl;

    /*int sum_kps = 0;
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
    cout<<"all kps "<<kps_1.size()<<" sum_kps "<<sum_kps<<endl;*/

    cv::Mat img1_kps, img2_kps;
    cv::drawKeypoints(img_1, kps_1, img1_kps);
    cv::drawKeypoints(img_2, kps_2, img2_kps);
    cv::imshow("img1_kps", img1_kps);
    cv::imshow("img2_kps", img2_kps);
    cv::waitKey(0);

    //接下来测试邻域功能
    //首先测试19的邻域,这个邻域比较好,特征点也比较多

    int index = 17;//18,37,41
    vector<cv::KeyPoint *> sub_ptrkps = gridNet.gridKeypoints[index];
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
    cout<<"no "<<index <<" has "<<gridNet. neighbors[index].size()<<" neighbors!"<<endl;
    for(int i=0; i<gridNet.neighbors[index].size(); i++){
        cout<<"grid keypoints size1  "<<gridNet.neighbors[index][i]->size()<<endl;
        for(int j=0; j<gridNet.neighbors[index][i]->size(); j++){
            cv::KeyPoint kp;
            kp.pt = (gridNet.neighbors[index][i]->at(j)->pt);
            sub_neighborkps.push_back(kp);
        }
        //size是行*列
        cout<<"desc size 1"<<gridNet.descs_neighbors[index][i].size<<endl;
    }
    cout<<"sub neighbor_kps size1 "<<sub_neighborkps.size()<<endl;
    cv::Mat img_subneib;
    cv::drawKeypoints(img_1, sub_neighborkps, img_subneib);
    cv::imshow("cur_kps1", img_subkps);
    cv::imshow("neighbor1", img_subneib);

    vector<cv::KeyPoint *> sub_ptrkps2 = gridNet2.gridKeypoints[index];
    vector<cv::KeyPoint> subkps2(sub_ptrkps2.size());
    //在什么地方重复的添加了特征点?
    for (int i = 0;i < subkps2.size(); i++) {
        subkps2[i].pt = sub_ptrkps2[i]->pt;
    }
    cv::Mat img_subkps2;
    cv::drawKeypoints(img_2, subkps2, img_subkps2);
    vector<cv::KeyPoint> sub_neighborkps2;
    sub_neighborkps2.reserve(300);
    cout<<"no "<<index <<" has "<<gridNet2. neighbors[index].size()<<" neighbors!"<<endl;
    for(int i=0; i<gridNet2.neighbors[index].size(); i++){
        cout<<"grid keypoints size 2 "<<gridNet2.neighbors[index][i]->size()<<endl;
        for(int j=0; j<gridNet2.neighbors[index][i]->size(); j++){
            cv::KeyPoint kp;
            kp.pt = (gridNet2.neighbors[index][i]->at(j)->pt);
            sub_neighborkps2.push_back(kp);
        }
        //size是行*列
        cout<<"desc size 2"<<gridNet2.descs_neighbors[index][i].size<<endl;
    }
    cout<<"sub neighbor_kps2 size "<<sub_neighborkps2.size()<<endl;
    cv::Mat img_subneib2;
    cv::drawKeypoints(img_2, sub_neighborkps2, img_subneib2);
    cv::imshow("cur_kps2", img_subkps2);
    cv::imshow("neighbor2", img_subneib2);
    cv::waitKey(0);

    // 好完成了,像素特征点提取,并且进行了划分,接下来进行数据匹配!!!
    // 进行特征点的匹配!
    SpaceNear_Matcher spacenear_matcher(&gridNet, &gridNet2);
    std::vector<cv::DMatch> sub_matches;
    spacenear_matcher.SpaceNearMatcher(index, sub_matches);
    cout<<"sub matches size "<<sub_matches.size()<<endl;

    vector<cv::KeyPoint> sub_kps1, sub_kps2;
    spacenear_matcher.drawMatches(img_1, sub_kps1, img_2, sub_kps2, sub_matches);
    cv::destroyAllWindows();

    vector<cv::DMatch> all_matches;
    vector<cv::KeyPoint> keyps_1, keyps_2;
    spacenear_matcher.SpaceNearMatcher(keyps_1, keyps_2, all_matches);
    runImagePair(img_1, img_2, sub_kps1, sub_kps2, sub_matches);
    cout<<"k1 - "<<keyps_1.size()<<" k2 - "<<keyps_2.size()<<" m - "<<all_matches.size()<<endl;
    cv::Mat img1_all_kps, img2_all_kps;
    cv::drawKeypoints(img_1, keyps_1, img1_all_kps);
    cv::drawKeypoints(img_2, keyps_2, img2_all_kps);
    cv::imshow("all_1",img1_all_kps);
    cv::imshow("all_2",img2_all_kps);
    cv::waitKey(0);
    cv::destroyAllWindows();
    //对keypoints_2 进行操作,删掉重复的点,并把相应的位置信息,重新给all_matches赋值
    /*
     *
     * ----------------     筛选点     -----------------
     *
     **/

    int preSize = kps_2.size();
    KeyPoints_Map kps_map(preSize);
    /*注意这两个功能必须得是同时使用的,一定对相同的数据进行操作*/
    kps_map.CreateUnorder_Map_KpIndex_vecIndex_IndexKP(kps_2);
    kps_map.CreateUnorder_Map_OuterIndex_InnerIndex(keyps_2);
    map<int, int>* indexMap = kps_map.GetIndexMap();
    for(int vm_index=0; vm_index<all_matches.size(); vm_index++){
        int outer_index = all_matches[vm_index].trainIdx;
        if(indexMap->find(outer_index) == indexMap->end()){
            cerr<<"some points can not find!"<<endl;
        }
        else{
            all_matches[vm_index].trainIdx = (*indexMap)[outer_index];
        }
    }
    /*
     *
     * !!!!!!!!!!!!!                   !!!!!!!!!!!!!!!
     *
     * */
    cout<<"###############################"<<endl;

    for(int i=0; i<all_matches.size(); i++){
        cout<<all_matches[i].trainIdx<<"     "<<all_matches[i].queryIdx<<endl;
    }
    cout<<keyps_1.size()<<endl;
    cv::Mat image;
    cv::drawMatches(img_1, keyps_1, img_2, kps_2, all_matches, image);
    cv::imshow("show", image);
    cv::waitKey(0);
    runImagePair(img_1, img_2, keyps_1, kps_2, all_matches);
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

void GmsMatch(Mat &img1, Mat &img2, vector<cv::KeyPoint>& kp1, vector<cv::KeyPoint>& kp2,
    vector<cv::DMatch>& matches_all) {

    vector<DMatch> matches_gms;

    std::chrono::system_clock::time_point start_gms = std::chrono::system_clock::now();
    // GMS filter
    std::vector<bool> vbInliers;
    gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);

    // collect matches
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches_all[i]);
        }
    }
    std::chrono::system_clock::time_point end_gms = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_gms = end_gms - start_gms;

    //just one or two sub areas is too small to do the matches
    cout << "Get total " << num_inliers << " matches." << endl;
    cout<<"GMS time out "<< elapsed_seconds_gms.count()<<"s."<<endl;
    // draw matching
    Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
    imshow("show", show);
    waitKey();
}
void ShowRotationMatchesImage(cv::Mat& img_1, vector<cv::KeyPoint>& kps_1,
                              cv::Mat& img_2, vector<cv::KeyPoint>& kps_2,
                              vector<cv::DMatch>& matches);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
    const int height = max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));

    Mat three_channel_1 = Mat::zeros(src1.rows, src1.cols, CV_8UC3);
    vector<Mat> channels_1;
    for (int i=0;i<3;i++)
    {
        channels_1.push_back(src1);
    }
    merge(channels_1,three_channel_1);

    Mat three_channel_2 = Mat::zeros(src2.rows, src2.cols, CV_8UC3);
    vector<Mat> channels_2;
    for (int i=0;i<3;i++)
    {
        channels_2.push_back(src2);
    }
    merge(channels_2,three_channel_2);

    three_channel_1.copyTo(output(Rect(0, 0, three_channel_1.cols, three_channel_1.rows)));
    three_channel_2.copyTo(output(Rect(three_channel_1.cols, 0, three_channel_2.cols, three_channel_2.rows)));

    if (type == 1)
    {
        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)three_channel_1.cols, 0.f));
            line(output, left, right, Scalar(0, 255, 255));
        }
    }
    else if (type == 2)
    {
        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)three_channel_1.cols, 0.f));
            line(output, left, right, Scalar(255, 0, 0));
        }

        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)three_channel_1.cols, 0.f));
            circle(output, left, 1, Scalar(0, 255, 255), 2);
            circle(output, right, 1, Scalar(0, 255, 0), 2);
        }
    }
    ShowRotationMatchesImage(src1, kpt1, src2, kpt2, inlier);
    return output;
}


void ShowRotationMatchesImage(cv::Mat& img_1, vector<cv::KeyPoint>& kps_1,
                              cv::Mat& img_2, vector<cv::KeyPoint>& kps_2,
                              vector<cv::DMatch>& matches){
    //将img逆时针旋转90°,特征点的x,y交换位置
    cv::Mat r90_img_1, r90_img_2;
    transpose(img_1, r90_img_1);
    transpose(img_2, r90_img_2);

    vector<cv::KeyPoint> transpose_kps_1, transpose_kps_2;
    transpose_kps_1.resize(kps_1.size());
    transpose_kps_2.resize(kps_2.size());
    for(int i=0; i<kps_1.size(); i++){
        transpose_kps_1[i].pt.x = kps_1[i].pt.y;
        transpose_kps_1[i].pt.y = kps_1[i].pt.x;
    }

    for(int i=0; i<kps_2.size(); i++){
        transpose_kps_2[i].pt.x = kps_2[i].pt.y;
        transpose_kps_2[i].pt.y = kps_2[i].pt.x;
    }
    cv::Mat out;
    cv::drawMatches(r90_img_1, transpose_kps_1, r90_img_2, transpose_kps_2, matches, out);
    transpose(out, out);
    cv::imshow("out", out);
    cv::waitKey(0);

}