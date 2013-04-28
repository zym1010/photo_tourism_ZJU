//
//  helper.h
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#ifndef __photo_tourism_core_code__helper__
#define __photo_tourism_core_code__helper__

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);

std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches);

void open_imgs_dir(const char *dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names, double downscale_factor); // now it will not work under Windows.

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2);

#endif /* defined(__photo_tourism_core_code__helper__) */
