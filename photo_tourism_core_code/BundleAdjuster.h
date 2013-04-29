//
//  BundleAdjuster.h
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-29.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#ifndef __photo_tourism_core_code__BundleAdjuster__
#define __photo_tourism_core_code__BundleAdjuster__

#include <opencv2/opencv.hpp>
#include "helper.h"

class BundleAdjuster {
public:
	void adjustBundle(std::vector<CloudPoint>& pointcloud,
					  cv::Mat& cam_matrix,
					  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
					  std::map<int ,cv::Matx34d>& Pmats);
private:
	int Count2DMeasurements(const std::vector<CloudPoint>& pointcloud);
};

#endif /* defined(__photo_tourism_core_code__BundleAdjuster__) */
