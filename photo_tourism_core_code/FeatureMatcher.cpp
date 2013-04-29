//
//  FeatureMatcher.cpp
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#include "FeatureMatcher.h"
#include "global.h"
#include <sstream>
#include <cassert>
using namespace std;
using namespace cv;

FeatureMatcher::FeatureMatcher(std::vector<cv::Mat>& imgs_,
                               std::vector<std::vector<cv::KeyPoint> >& imgpts_):
imgs(imgs_), imgpts(imgpts_){
    //ZYM: create keypoint detector and keypoint description extractor
	detector = cv::FeatureDetector::create("PyramidFAST");//PyramidFAST detector
	extractor = cv::DescriptorExtractor::create("ORB");//ORB descriptor
    
    detector->detect(imgs, imgpts);//ZYM: fill imgpts with keypoints of each image. imgpts is a list of lists of keypoints (of type 'cv::KeyPoint'); imgs is not modified, and imgpts is filled up.
    
	extractor->compute(imgs, imgpts, descriptors);//ZYM: here, imgpts and descriptors are NON const, so imgpts can be modified to match the number of descriptors.
    //descriptors has N rows where N is the size of imgpts, here, using ORB feature, each row has 32 elements (descriptors has 32 cols)
    
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write descriptors to file begins" << endl;
        cv::FileStorage f;
        f.open(descriptorDebugOutput, cv::FileStorage::WRITE);
        
        for (unsigned i = 0; i < descriptors.size(); i++) {
            stringstream ss;
            ss << "mat" << i; //names must begin with a letter
            cerr << ss.str() << endl;
            f << ss.str() << descriptors[i];
        }
        f.release();
        cerr << "write descriptors to file ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
}

void FeatureMatcher::MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches){
    const Mat& descriptors_1 = descriptors[idx_i]; //ZYM: alias for (gray) image descriptors 1
    const Mat& descriptors_2 = descriptors[idx_j]; //ZYM: alias for (gray) image descriptors 2
    
    std::vector< DMatch > good_matches_;//here, I should PRESERVE the match!
    
    //matching descriptor vectors using Brute Force matcher
    BFMatcher matcher(NORM_HAMMING,true); //allow cross-check. use Hamming distance for binary descriptor (ORB)
    
    matcher.match( descriptors_1, descriptors_2, *matches);
    assert(matches->size() > 0);
}