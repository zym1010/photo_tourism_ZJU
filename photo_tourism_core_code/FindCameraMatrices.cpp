//
//  FindCameraMatrices.cpp
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#include "FindCameraMatrices.h"
#include "helper.h"
#include <cassert>
using namespace std;
using namespace cv;

cv::Mat GetFundamentalMat(const std::vector<cv::KeyPoint>& imgpts1,
                          const std::vector<cv::KeyPoint>& imgpts2,
                          std::vector<cv::KeyPoint>& imgpts1_good,
                          std::vector<cv::KeyPoint>& imgpts2_good,
                          std::vector<cv::DMatch>& matches
						  ){
    vector<uchar> status;
    
    //ZYM: again, here these two variables are cleared.
	imgpts1_good.clear(); imgpts2_good.clear();
    
    //ZYM: in this function, we're manipulating actually the _tmp version.
	vector<KeyPoint> imgpts1_tmp;
	vector<KeyPoint> imgpts2_tmp;
    
    GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
    
    //ZYM: what's really interesting starts...
    Mat F;
    
    {
        vector<Point2f> pts1,pts2;
        //ZYM: convert keypoints to points.
		KeyPointsToPoints(imgpts1_tmp, pts1);
		KeyPointsToPoints(imgpts2_tmp, pts2);
        
        
        double minVal,maxVal;
		cv::minMaxIdx(pts1,&minVal,&maxVal);//ZYM: seems to flatten the pts1 beforehand...
        //ZYM: but I don't want to care about this...
        //ZYM: minMaxIdx in theory works on 1 dimension things.
        //ZYM: problems: 1. using vector of 2d points may not be legal; 2. this may not be correct, according to Snavely. may be (maxVal-minVal) is better?
        
        F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
        //ZYM: here, status is output, so what's in it before is not important.
    }
    
    vector<DMatch> new_matches;
    
#ifdef PHOTO_TOURISM_DEBUG
    cerr << "======DEBUGGING INFO BEGINS======" << endl;
    cerr << "F keeping " << cv::countNonZero(status) << " / " << status.size() << endl;
    cerr << "======DEBUGGING INFO ENDS======" << endl;
#endif
    
    for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) //ZYM: this pair of match is counted as good match
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);
            
            new_matches.push_back(matches[i]);//ZYM: this is correct... I don't know what the other fork means.
		}
	}
    
    assert(new_matches.size()==cv::countNonZero(status));
    assert(matches.size()==status.size());
    
    matches = new_matches;
    return F;
}