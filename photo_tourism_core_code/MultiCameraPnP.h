//
//  MultiCameraPnP.h
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#ifndef __photo_tourism_core_code__MultiCameraPnP__
#define __photo_tourism_core_code__MultiCameraPnP__
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include "FeatureMatcher.h"

class MultiCameraPnP {
    
public:
    //I dropped the last argument for path, since it's used for reading camera parameters from file, but we don't have such a file.
    MultiCameraPnP(const std::vector<cv::Mat>& imgs_,
                   const std::vector<std::string>& imgs_names_);
    void RecoverDepthFromImages();
private:
    //private functions
    void OnlyMatchFeatures();
    void PruneMatchesBasedOnF();
    
    
    
    //private variables
    std::vector<cv::Mat> imgs;
	std::vector<std::string> imgs_names;
    std::vector<cv::Mat_<cv::Vec3b> > imgs_orig;
    
    std::vector<std::vector<cv::KeyPoint> > imgpts_good;
    std::vector<std::vector<cv::KeyPoint> > imgpts;
    
    //don't know if all these are useful.
    cv::Mat K;
	cv::Mat_<double> Kinv;
    
    cv::Mat cam_matrix,distortion_coeff;
	cv::Mat distcoeff_32f;
	cv::Mat K_32f;
    
    std::map<std::pair<int,int> ,std::vector<cv::DMatch> > matches_matrix;

    cv::Ptr<FeatureMatcher> feature_matcher;
    

    
};

#endif /* defined(__photo_tourism_core_code__MultiCameraPnP__) */
