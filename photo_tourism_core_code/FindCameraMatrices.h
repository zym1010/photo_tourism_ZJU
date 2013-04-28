//
//  FindCameraMatrices.h
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#ifndef __photo_tourism_core_code__FindCameraMatrices__
#define __photo_tourism_core_code__FindCameraMatrices__

#include <opencv2/opencv.hpp>

cv::Mat GetFundamentalMat(const std::vector<cv::KeyPoint>& imgpts1,
                          const std::vector<cv::KeyPoint>& imgpts2,
                          std::vector<cv::KeyPoint>& imgpts1_good,
                          std::vector<cv::KeyPoint>& imgpts2_good,
                          std::vector<cv::DMatch>& matches
						  );

#endif /* defined(__photo_tourism_core_code__FindCameraMatrices__) */

