//
//  MultiCameraPnP.cpp
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#include "MultiCameraPnP.h"
#include "global.h"
#include <sstream>


using namespace std;

MultiCameraPnP::MultiCameraPnP(const std::vector<cv::Mat>& imgs_,
                               const std::vector<std::string>& imgs_names_):imgs_names(imgs_names_)
{
    //normalize images begins
    for (unsigned int i=0; i<imgs_.size(); i++) {
		imgs_orig.push_back(cv::Mat_<cv::Vec3b>()); //cv::Vec3b means 3 channels of bytes
		if (!imgs_[i].empty()) {
			if (imgs_[i].type() == CV_8UC1) {
				cvtColor(imgs_[i], imgs_orig[i], CV_GRAY2BGR);
			} else if (imgs_[i].type() == CV_32FC3 || imgs_[i].type() == CV_64FC3) {
				imgs_[i].convertTo(imgs_orig[i],CV_8UC3,255.0);
			} else {
				imgs_[i].copyTo(imgs_orig[i]);
			}
		}
		
		imgs.push_back(cv::Mat());
		cvtColor(imgs_orig[i],imgs[i], CV_BGR2GRAY);
		
		imgpts.push_back(std::vector<cv::KeyPoint>());
		imgpts_good.push_back(std::vector<cv::KeyPoint>());
	}
    
#ifdef PHOTO_TOURISM_DEBUG
    cerr << "======DEBUGGING INFO BEGINS======" << endl;
    cerr << "write RGB images to file begins" << endl;
    for (unsigned i = 0; i < imgs_orig.size(); i++) {
        cv::imwrite("RGB"+imgs_names[i], imgs_orig[i]);
    }
    cerr << "write RGB images to file ends" << endl;
    
    cerr << "write Gray images to file begins" << endl;
    for (unsigned i = 0; i < imgs.size(); i++) {
        cv::imwrite("Gray"+imgs_names[i], imgs[i]);
    }
    cerr << "write Gray images to file ends" << endl;
    
    cerr << "======DEBUGGING INFO ENDS======" << endl;
#endif
    
    //normalize images ends
    
    
    
    //initialize camera matrices begins
    cv::Size imgs_size = imgs_[0].size();
    double max_w_h = MAX(imgs_size.height,imgs_size.width);
    cam_matrix = (cv::Mat_<double>(3,3) <<	max_w_h ,	0	,		imgs_size.width/2.0,
                  0,			max_w_h,	imgs_size.height/2.0,
                  0,			0,			1);
    distortion_coeff = cv::Mat_<double>::zeros(1,4);
    
    K = cam_matrix; //ZYM: K is camera matrix (instrinsic matrix)
	invert(K, Kinv); //ZYM: get inverse of camera matrix
    
	distortion_coeff.convertTo(distcoeff_32f,CV_32FC1); //ZYM: distcoeff float version
	K.convertTo(K_32f,CV_32FC1); //ZYM: float version
    
#ifdef PHOTO_TOURISM_DEBUG
    cerr << "======DEBUGGING INFO BEGINS======" << endl;
    cerr << "cam_matrix: " << K << endl;
    cerr << "cam_matrix inverted: " << Kinv << endl;
    cerr << "======DEBUGGING INFO ENDS======" << endl;
#endif
    
    //initialize camera matrices ends
    
    
}


void MultiCameraPnP::RecoverDepthFromImages(){
    OnlyMatchFeatures();
}

void MultiCameraPnP::OnlyMatchFeatures(){
    feature_matcher = new FeatureMatcher(imgs,imgpts);
}
