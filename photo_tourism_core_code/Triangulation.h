//
//  Triangulation.h
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-29.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#ifndef __photo_tourism_core_code__Triangulation__
#define __photo_tourism_core_code__Triangulation__

#include <opencv2/opencv.hpp>
#include "helper.h"


double TriangulatePoints(const std::vector<cv::KeyPoint>& pt_set1,
                         const std::vector<cv::KeyPoint>& pt_set2,
                         const cv::Mat& K,
                         const cv::Mat& Kinv,
                         const cv::Mat& distcoeff,
                         const cv::Matx34d& P,
                         const cv::Matx34d& P1,
                         std::vector<CloudPoint>& pointcloud,
                         std::vector<cv::KeyPoint>& correspImg1Pt);


/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
                                                cv::Matx34d P,			//camera 1 matrix
                                                cv::Point3d u1,			//homogenous image point in 2nd camera
                                                cv::Matx34d P1			//camera 2 matrix
                                                );


#endif /* defined(__photo_tourism_core_code__Triangulation__) */
