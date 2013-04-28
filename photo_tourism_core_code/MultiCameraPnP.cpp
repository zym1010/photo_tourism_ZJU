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
#include "helper.h"
#include "FindCameraMatrices.h"


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
    
    PruneMatchesBasedOnF();
}

void MultiCameraPnP::OnlyMatchFeatures(){
    feature_matcher = new FeatureMatcher(imgs,imgpts);
    
    //prepare to calculate matches between every pair of images.
    int loop1_top = (int)imgs.size() - 1, loop2_top = (int)imgs.size();
#pragma omp parallel for
    //ZYM: frame_num_i and frame_num_j are indecies for 2 images currently being matched.
    //ZYM: for N images, C(N,2) inner loops are executed in total.
    
    for (int frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
        for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
        {
            std::vector<cv::DMatch> matches_tmp; //ZYM: temporary variable saving matches.
            feature_matcher->MatchFeatures(frame_num_i,frame_num_j,&matches_tmp);//ZYM: get matches. this match is not the match filtered by RANSAC, just the NN match.
            matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;//ZYM: create a match matrix.
            //good matches are NOT preserved!!!
            
            std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
            matches_matrix[std::make_pair(frame_num_j,frame_num_i)] = matches_tmp_flip; //ZYM: create flip version to make the matches_matrix complete.
        }
    }
    
#ifdef PHOTO_TOURISM_DEBUG
    cerr << "======DEBUGGING INFO BEGINS======" << endl;
    cerr << "write match matrix to file begins" << endl;
    cv::FileStorage f;
    f.open(initialMatchMatrixDebugOutput, cv::FileStorage::WRITE);
    f << "test_list" << "[";
    for (unsigned i = 0; i < imgs.size()-1; i++) {
        for (unsigned j = i+1; j < imgs.size(); j++) {
            const std::vector<cv::DMatch> & current_match_vector = matches_matrix[std::make_pair(i,j)];
            for (unsigned k = 0; k < current_match_vector.size(); k++) {
                f << current_match_vector[k].queryIdx;
                f << current_match_vector[k].trainIdx;
            }
        }
    }
    f << "]";
    f.release();
    cerr << "write match matrix to file ends" << endl;
    cerr << "======DEBUGGING INFO ENDS======" << endl;
#endif
    
}


void MultiCameraPnP::PruneMatchesBasedOnF() {
	//prune the match between <_i> and all views using the Fundamental matrix to prune
#pragma omp parallel for
	for (int _i=0; _i < imgs.size() - 1; _i++)
	{
		for (unsigned int _j=_i+1; _j < imgs.size(); _j++) {
			int older_view = _i, working_view = _j;
            
			GetFundamentalMat( imgpts[older_view],
                              imgpts[working_view],
                              imgpts_good[older_view],
                              imgpts_good[working_view],
                              matches_matrix[std::make_pair(older_view,working_view)]
                              );
			//update flip matches as well
#pragma omp critical
			matches_matrix[std::make_pair(working_view,older_view)] = FlipMatches(matches_matrix[std::make_pair(older_view,working_view)]);
		}
	}
#ifdef PHOTO_TOURISM_DEBUG
    cerr << "======DEBUGGING INFO BEGINS======" << endl;
    cerr << "write match matrix to file begins" << endl;
    cv::FileStorage f;
    f.open(refinedMatchMatrixDebugOutput, cv::FileStorage::WRITE);
    f << "test_list" << "[";
    for (unsigned i = 0; i < imgs.size()-1; i++) {
        for (unsigned j = i+1; j < imgs.size(); j++) {
            const std::vector<cv::DMatch> & current_match_vector = matches_matrix[std::make_pair(i,j)];
            for (unsigned k = 0; k < current_match_vector.size(); k++) {
                f << current_match_vector[k].queryIdx;
                f << current_match_vector[k].trainIdx;
            }
        }
    }
    f << "]";
    f.release();
    cerr << "write match matrix to file ends" << endl;
    cerr << "======DEBUGGING INFO ENDS======" << endl;
#endif
    
}
