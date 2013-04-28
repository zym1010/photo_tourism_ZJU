//
//  FeatureMatcher.h
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#ifndef __photo_tourism_core_code__FeatureMatcher__
#define __photo_tourism_core_code__FeatureMatcher__

#include <vector>
#include <opencv2/opencv.hpp>

class FeatureMatcher {
private:
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	std::vector<cv::Mat> descriptors;
	std::vector<cv::Mat>& imgs;
	std::vector<std::vector<cv::KeyPoint> >& imgpts;
public:
    FeatureMatcher(std::vector<cv::Mat>& imgs,
					   std::vector<std::vector<cv::KeyPoint> >& imgpts);
	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches);
};

#endif /* defined(__photo_tourism_core_code__FeatureMatcher__) */
