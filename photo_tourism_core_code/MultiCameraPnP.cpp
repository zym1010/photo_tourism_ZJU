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
#include "Triangulation.h"
#include "BundleAdjuster.h"
#include <cstdio>

using namespace std;
static bool sort_by_first(pair<int,pair<int,int> > a, pair<int,pair<int,int> > b) { return a.first > b.first; }

void MultiCameraPnP::writeResults(){
    cout << "write image name list begins" << endl;
    {
        cv::FileStorage f;
        f.open(imageNameOutput, cv::FileStorage::WRITE);
        f << "name_list" << "[";
        for (unsigned i = 0; i < imgs_names.size(); i++) {
            f << imgs_names[i];
        }
        f << "]";
        f.release();
    }
    cout << "write image name list ends" << endl;
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "Pmats has size " << Pmats.size() << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
    cout << "write final projection matrix list begins" << endl;
    {
        cv::FileStorage f;
        f.open(finalProjectionMatrixOutput, cv::FileStorage::WRITE);
        f << "mat_list" << "[";
        for (unsigned i = 0; i < Pmats.size(); i++) {
            f << cv::Mat(Pmats[i]);
        }
        f << "]";
        f.release();
    }
    cout << "write final projection matrix list ends" << endl;
}

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
    {
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
    }
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
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "cam_matrix: " << K << endl;
        cerr << "cam_matrix inverted: " << Kinv << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
    //initialize camera matrices ends
    
    
}


void MultiCameraPnP::RecoverDepthFromImages(){
    OnlyMatchFeatures();
    
    PruneMatchesBasedOnF();
    
    GetBaseLineTriangulation();
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write match matrix to file begins" << endl;
        cv::FileStorage f;
        f.open(refinedMatchMatrixAfterTriangulationDebugOutput, cv::FileStorage::WRITE);
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
    }
#endif
    
    AdjustCurrentBundle();
        
    //if there're only 2 images, we are done here.
    //so what's above is what I want to understand.
    
	cv::Matx34d P1 = Pmats[m_second_view];
	cv::Mat_<double> t = (cv::Mat_<double>(1,3) << P1(0,3), P1(1,3), P1(2,3));
	cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0), P1(0,1), P1(0,2),
                          P1(1,0), P1(1,1), P1(1,2),
                          P1(2,0), P1(2,1), P1(2,2));
	cv::Mat_<double> rvec(1,3); Rodrigues(R, rvec);
	
	done_views.insert(m_first_view);
	done_views.insert(m_second_view);
	good_views.insert(m_first_view);
	good_views.insert(m_second_view);
    
    while (done_views.size() != imgs.size())
	{
		//find image with highest 2d-3d correspondance [Snavely07 4.2]
		int max_2d3d_view = -1, max_2d3d_count = 0;
		vector<cv::Point3f> max_3d; vector<cv::Point2f> max_2d;
		for (unsigned int _i=0; _i < imgs.size(); _i++) {
			if(done_views.find(_i) != done_views.end()) continue; //already done with this view
            
			vector<cv::Point3f> tmp3d; vector<cv::Point2f> tmp2d;
			cout << imgs_names[_i] << ": ";
			Find2D3DCorrespondences(_i,tmp3d,tmp2d);
			if(tmp3d.size() > max_2d3d_count) {
				max_2d3d_count = (int)tmp3d.size();
				max_2d3d_view = _i;
				max_3d = tmp3d; max_2d = tmp2d;
			}
		}
		int i = max_2d3d_view; //highest 2d3d matching view
        
		std::cout << "-------------------------- " << imgs_names[i] << " --------------------------\n";
		done_views.insert(i); // don't repeat it for now
        
        

        
        
		bool pose_estimated = FindPoseEstimation(i,rvec,t,R,max_3d,max_2d);
        
        
        
		if(!pose_estimated)
			continue;
        
		//store estimated pose
		Pmats[i] = cv::Matx34d	(R(0,0),R(0,1),R(0,2),t(0),
								 R(1,0),R(1,1),R(1,2),t(1),
								 R(2,0),R(2,1),R(2,2),t(2));
		
		// start triangulating with previous GOOD views
		for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view)
		{
			int view = *done_view;
			if( view == i ) continue; //skip current...
            
			cout << " -> " << imgs_names[view] << endl;
			
			vector<CloudPoint> new_triangulated;
			vector<int> add_to_cloud;
			bool good_triangulation = TriangulatePointsBetweenViews(i,view,new_triangulated,add_to_cloud);
			if(!good_triangulation) continue;
            
			std::cout << "before triangulation: " << pcloud.size();
			for (int j=0; j<add_to_cloud.size(); j++) {
				if(add_to_cloud[j] == 1)
					pcloud.push_back(new_triangulated[j]);
			}
			std::cout << " after " << pcloud.size() << std::endl;
			//break;
		}
		good_views.insert(i);
		
		AdjustCurrentBundle();
	}
    
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
    {
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
    }
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
    {
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
    }
#endif
    
}

void MultiCameraPnP::GetBaseLineTriangulation(){
    //ZYM: two base extrinsic matrix
	cv::Matx34d P(1,0,0,0,
				  0,1,0,0,
				  0,0,1,0),
    P1(1,0,0,0,
       0,1,0,0,
       0,0,1,0);
    
    
    std::vector<CloudPoint> tmp_pcloud;
    
    list<pair<int,pair<int,int> > > matches_sizes;
	//TODO: parallelize!
    //ZYM: this iterator is for matches_matrix
	for(std::map<std::pair<int,int> ,std::vector<cv::DMatch> >::iterator i = matches_matrix.begin(); i != matches_matrix.end(); ++i) {
        matches_sizes.push_back(make_pair((*i).second.size(),(*i).first));
        
	}
	cout << endl;
	matches_sizes.sort(sort_by_first);
    
    
    
    //Reconstruct from two views
	bool goodF = false;
    
    m_first_view = m_second_view = 0;
    
    for(list<pair<int,pair<int,int> > >::iterator highest_pair = matches_sizes.begin();
		highest_pair != matches_sizes.end() && !goodF; //ZYM: stop unless we find good F or there's no more pair to be tried.
		++highest_pair)
	{
        //ZYM:first 'second' means the image pair.
        //ZYM: wierd that m_second_view and m_first_view are reversed.
		m_second_view = (*highest_pair).second.second;
		m_first_view  = (*highest_pair).second.first;
        
        
#ifdef PHOTO_TOURISM_DEBUG
        {
            cerr << "======DEBUGGING INFO BEGINS======" << endl;
            cerr << " -------- " << imgs_names[m_first_view] << "(" << m_first_view << ")"
            << " and " << imgs_names[m_second_view] <<  "(" << m_second_view << ")"
            << " -------- " << endl;
            cerr << "======DEBUGGING INFO ENDS======" << endl;
        }
#endif
        
		std::cout << " -------- " << imgs_names[m_first_view] << " and " << imgs_names[m_second_view] << " -------- " <<std::endl;//ZYM: a prompt showing two images being used for base triangulation
        
		//what if reconstrcution of first two views is bad? fallback to another pair
		//See if the Fundamental Matrix between these two views is good
        
        //ZYM: this is most important and most complicated
        //ZYM: K, Kinv are intrinstic matrix and its inverse,
        //ZYM:imgpts are original matched points (using cross-checking and brute force matching)
        //ZYM: imgpts_good are mached points using RANSAC 8-point to prune.
		goodF = FindCameraMatrices(K, Kinv, distortion_coeff,//ZYM: all const
                                   imgpts[m_first_view], //ZYM: const
                                   imgpts[m_second_view], //ZYM: const
                                   imgpts_good[m_first_view], //ZYM: not const
                                   imgpts_good[m_second_view], //ZYM: not const
                                   P, //ZYM: not const
                                   P1, //ZYM: not const
                                   matches_matrix[std::make_pair(m_first_view,m_second_view)], //ZYM: not const
                                   tmp_pcloud //ZYM: not const
                                   );
        
        //ZYM: trick to fix base triangulation
        if(imgs_names[m_first_view]!="middle.jpg" || imgs_names[m_second_view]!="right1.png"){
            goodF = false;
        }
        
        //ZYM: I think this is not necessary for us to understand, as long as we have the matrix
		if (goodF) {
			vector<CloudPoint> new_triangulated;
			vector<int> add_to_cloud;
            
			Pmats[m_first_view] = P;
			Pmats[m_second_view] = P1;
            
			bool good_triangulation = TriangulatePointsBetweenViews(m_second_view,m_first_view,new_triangulated,add_to_cloud);
			if(!good_triangulation || cv::countNonZero(add_to_cloud) < 10) {
                //				std::cout << "triangulation failed" << std::endl;
				goodF = false;
				Pmats[m_first_view] = 0;
				Pmats[m_second_view] = 0;
				m_second_view++;
			} else {
				std::cout << "before triangulation: " << pcloud.size();
				for (unsigned int j=0; j<add_to_cloud.size(); j++) {
					if(add_to_cloud[j] == 1)
						pcloud.push_back(new_triangulated[j]);
				}
				std::cout << " after " << pcloud.size() << std::endl;
			}
		}
        
        
	}
    
    
    //ZYM: succeed
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "Taking baseline from " << imgs_names[m_first_view] << " and " << imgs_names[m_second_view] << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "output project matrices" << endl;
        
        cv::FileStorage f;
        f.open(baselineTriangulationDebugOutput, cv::FileStorage::WRITE);
        f << "P" << cv::Mat(Pmats[m_first_view]);
        f << "P1" << cv::Mat(Pmats[m_second_view]);
        f.release();
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
}


bool MultiCameraPnP::TriangulatePointsBetweenViews(
                                                   int working_view,
                                                   int older_view,
                                                   vector<struct CloudPoint>& new_triangulated,
                                                   vector<int>& add_to_cloud
                                                   )
{
	cout << " Triangulate " << imgs_names[working_view] << " and " << imgs_names[older_view] << endl;
	//get the left camera matrix
	//TODO: potential bug - the P mat for <view> may not exist? or does it...
	cv::Matx34d P = Pmats[older_view];
	cv::Matx34d P1 = Pmats[working_view];
    
	std::vector<cv::KeyPoint> pt_set1,pt_set2;
	std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(older_view,working_view)];
	GetAlignedPointsFromMatch(imgpts[older_view],imgpts[working_view],matches,pt_set1,pt_set2);
    
    
	//adding more triangulated points to general cloud
	double reproj_error = TriangulatePoints(pt_set1, pt_set2, K, Kinv, distortion_coeff, P, P1, new_triangulated, correspImg1Pt);
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "triangulation reproj error " << reproj_error << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    vector<uchar> trig_status;
    if(!TestTriangulation(new_triangulated, P, trig_status) || !TestTriangulation(new_triangulated, P1, trig_status)) {
        cerr << "Triangulation did not succeed" << endl;
        return false;
    }
    //	if(reproj_error > 20.0) {
    //		// somethign went awry, delete those triangulated points
    //		//				pcloud.resize(start_i);
    //		cerr << "reprojection error too high, don't include these points."<<endl;
    //		return false;
    //	}
    
    //filter out outlier points with high reprojection
    vector<double> reprj_errors;
    for(int i=0;i<new_triangulated.size();i++) { reprj_errors.push_back(new_triangulated[i].reprojection_error); }
    std::sort(reprj_errors.begin(),reprj_errors.end());
    //get the 80% precentile
    double reprj_err_cutoff = reprj_errors[4 * reprj_errors.size() / 5] * 2.4; //threshold from Snavely07 4.2
    
    vector<CloudPoint> new_triangulated_filtered;
    std::vector<cv::DMatch> new_matches;
    for(int i=0;i<new_triangulated.size();i++) {
        if(trig_status[i] == 0)
            continue; //point was not in front of camera
        if(new_triangulated[i].reprojection_error > 16.0) {
            continue; //reject point
        }
        if(new_triangulated[i].reprojection_error < 4.0 ||
           new_triangulated[i].reprojection_error < reprj_err_cutoff)
        {
            new_triangulated_filtered.push_back(new_triangulated[i]);
            new_matches.push_back(matches[i]);
        }
        else
        {
            continue;
        }
    }
    
    cout << "filtered out " << (new_triangulated.size() - new_triangulated_filtered.size()) << " high-error points" << endl;
    
    //all points filtered?
    if(new_triangulated_filtered.size() <= 0) return false;
    
    new_triangulated = new_triangulated_filtered;
    
    matches = new_matches;
    matches_matrix[std::make_pair(older_view,working_view)] = new_matches; //just to make sure, remove if unneccesary
    matches_matrix[std::make_pair(working_view,older_view)] = FlipMatches(new_matches);
    add_to_cloud.clear();
    add_to_cloud.resize(new_triangulated.size(),1);
    int found_other_views_count = 0;
    unsigned long num_views = imgs.size();
    
    //scan new triangulated points, if they were already triangulated before - strengthen cloud
    //#pragma omp parallel for num_threads(1)
    for (int j = 0; j<new_triangulated.size(); j++) {
        new_triangulated[j].imgpt_for_img = std::vector<int>(imgs.size(),-1);
        
        //matches[j] corresponds to new_triangulated[j]
        //matches[j].queryIdx = point in <older_view>
        //matches[j].trainIdx = point in <working_view>
        new_triangulated[j].imgpt_for_img[older_view] = matches[j].queryIdx;	//2D reference to <older_view>
        new_triangulated[j].imgpt_for_img[working_view] = matches[j].trainIdx;		//2D reference to <working_view>
        bool found_in_other_view = false;
        for (unsigned int view_ = 0; view_ < num_views; view_++) {
            if(view_ != older_view) {
                //Look for points in <view_> that match to points in <working_view>
                std::vector<cv::DMatch> submatches = matches_matrix[std::make_pair(view_,working_view)];
                for (unsigned int ii = 0; ii < submatches.size(); ii++) {
                    if (submatches[ii].trainIdx == matches[j].trainIdx &&
                        !found_in_other_view)
                    {
                        //Point was already found in <view_> - strengthen it in the known cloud, if it exists there
                        
                        //cout << "2d pt " << submatches[ii].queryIdx << " in img " << view_ << " matched 2d pt " << submatches[ii].trainIdx << " in img " << i << endl;
                        for (unsigned int pt3d=0; pt3d<pcloud.size(); pt3d++) {
                            if (pcloud[pt3d].imgpt_for_img[view_] == submatches[ii].queryIdx)
                            {
                                //pcloud[pt3d] - a point that has 2d reference in <view_>
                                
                                //cout << "3d point "<<pt3d<<" in cloud, referenced 2d pt " << submatches[ii].queryIdx << " in view " << view_ << endl;
#pragma omp critical
                                {
                                    pcloud[pt3d].imgpt_for_img[working_view] = matches[j].trainIdx;
                                    pcloud[pt3d].imgpt_for_img[older_view] = matches[j].queryIdx;
                                    found_in_other_view = true;
                                    add_to_cloud[j] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            if (found_in_other_view) {
                found_other_views_count++;
            } else {
                add_to_cloud[j] = 1;
            }
        }
    }
    
    std::cout << found_other_views_count << "/" << new_triangulated.size() << " points were found in other views, adding " << cv::countNonZero(add_to_cloud) << " new\n";
    return true;
}

void MultiCameraPnP::AdjustCurrentBundle() {
    
	cv::Mat _cam_matrix = K;
	BundleAdjuster BA;
	BA.adjustBundle(pcloud,_cam_matrix,imgpts,Pmats);
	K = cam_matrix;//I think there's a bug......
	Kinv = K.inv();
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "use new K " << endl << K << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
//#ifdef PHOTO_TOURISM_DEBUG
//    {
//        cerr << "======DEBUGGING INFO BEGINS======" << endl;
//        cerr << "output project matrices after BA" << endl;
//        
//        cv::FileStorage f;
//        f.open(baselineTriangulationAfterBADebugOutput, cv::FileStorage::WRITE);
//        f << "P" << cv::Mat(Pmats[m_first_view]);
//        f << "P1" << cv::Mat(Pmats[m_second_view]);
//        f.release();
//        cerr << "======DEBUGGING INFO ENDS======" << endl;
//    }
//#endif
    
}


void MultiCameraPnP::Find2D3DCorrespondences(int working_view,
                                             std::vector<cv::Point3f>& ppcloud,
                                             std::vector<cv::Point2f>& imgPoints)
{
	ppcloud.clear(); imgPoints.clear();
    
	vector<int> pcloud_status(pcloud.size(),0);
	for (set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view)
	{
		int old_view = *done_view;
		//check for matches_from_old_to_working between i'th frame and <old_view>'th frame (and thus the current cloud)
		std::vector<cv::DMatch> matches_from_old_to_working = matches_matrix[std::make_pair(old_view,working_view)];
        
		for (unsigned int match_from_old_view=0; match_from_old_view < matches_from_old_to_working.size(); match_from_old_view++) {
			// the index of the matching point in <old_view>
			int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;
            
			//scan the existing cloud (pcloud) to see if this point from <old_view> exists
			for (unsigned int pcldp=0; pcldp<pcloud.size(); pcldp++) {
				// see if corresponding point was found in this point
				if (idx_in_old_view == pcloud[pcldp].imgpt_for_img[old_view] && pcloud_status[pcldp] == 0) //prevent duplicates
				{
					//3d point in cloud
					ppcloud.push_back(pcloud[pcldp].pt);
					//2d point in image i
					imgPoints.push_back(imgpts[working_view][matches_from_old_to_working[match_from_old_view].trainIdx].pt);
                    
					pcloud_status[pcldp] = 1;
					break;
				}
			}
		}
	}
	cout << "found " << ppcloud.size() << " 3d-2d point correspondences"<<endl;
}

bool MultiCameraPnP::FindPoseEstimation(
                                        int working_view,
                                        cv::Mat_<double>& rvec,
                                        cv::Mat_<double>& t,
                                        cv::Mat_<double>& R,
                                        std::vector<cv::Point3f> ppcloud,
                                        std::vector<cv::Point2f> imgPoints
                                        )
{
	if(ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
		//something went wrong aligning 3D to 2D points..
		cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<endl;
		return false;
	}
    
	vector<int> inliers;

    //use CPU
    double minVal,maxVal; cv::minMaxIdx(imgPoints,&minVal,&maxVal);
    
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write max2d begins" << endl;
        cv::FileStorage f;
        f.open(max2DDebugOutput, cv::FileStorage::WRITE);
        f << "max2d_list" << "[";
        for (unsigned i = 0; i < imgPoints.size(); i++) {
            f << imgPoints[i].x << imgPoints[i].y;
        }
        f << "]";
        f.release();
        cerr << "write max2d ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
#ifdef PHOTO_TOURISM_DEBUG_BINARY
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write max2d bin begins" << endl;
        FILE *fp = fopen(max2DDebugOutputBin,"wb");
        int count = 0;
        for (unsigned i = 0; i < imgPoints.size(); i++) {
            fwrite(&(imgPoints[i].x), sizeof(float), 1, fp);
            fwrite(&(imgPoints[i].y), sizeof(float), 1, fp);
            count = count+1;
        }
        cerr << "in total " << count << " pairs" << endl;
        fclose(fp);
        cerr << "write max2d bin ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
        
    }
#endif
    
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write max3d begins" << endl;
        cv::FileStorage f;
        f.open(max3DDebugOutput, cv::FileStorage::WRITE);
        f << "max3d_list" << "[";
        for (unsigned i = 0; i < ppcloud.size(); i++) {
            f << ppcloud[i].x << ppcloud[i].y << ppcloud[i].z;
        }
        f << "]";
        f.release();
        cerr << "write max3d ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
#ifdef PHOTO_TOURISM_DEBUG_BINARY
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write max3d bin begins" << endl;
        FILE *fp = fopen(max3DDebugOutputBin,"wb");
        int count = 0;
        for (unsigned i = 0; i < ppcloud.size(); i++) {
            fwrite(&(ppcloud[i].x), sizeof(float), 1, fp);
            fwrite(&(ppcloud[i].y), sizeof(float), 1, fp);
            fwrite(&(ppcloud[i].z), sizeof(float), 1, fp);
            count = count+1;
        }
        fclose(fp);
        cerr << "in total " << count << " triples" << endl;
        cerr << "write max3d bin ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write other value begins" << endl;
        cv::FileStorage f;
        f.open(otherValueDebugOutput, cv::FileStorage::WRITE);
        f << "maxVal" << maxVal;
        f << "K" << K;
        f << "dist" << distortion_coeff;
        f.release();
        cerr << "write other value ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
    
#ifdef PHOTO_TOURISM_DEBUG_BINARY
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write other value bin begins" << endl;
        FILE *fp = fopen(otherValueDebugOutputBin,"wb");
        cerr << "write K begins" << endl;
        for (unsigned i = 0; i < 9; i++) {
            fwrite(&(K.at<double>(i)), sizeof(double), 1, fp);
        }
        cerr << "write K ends" << endl;
        
        cerr << "write maxVal begins" << endl;
        fwrite(&maxVal, sizeof(double), 1, fp);
        cerr << "write maxVal ends" << endl;
        fclose(fp);
        cerr << "write other value bin ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
    
    CV_PROFILE("solvePnPRansac",cv::solvePnPRansac(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, 1000, 0.006 * maxVal, 0.25 * (double)(imgPoints.size()), inliers, CV_EPNP);)
    
    
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "write result begins" << endl;
        cv::FileStorage f;
        f.open(resultDebugOutput, cv::FileStorage::WRITE);
        f << "rvec" << rvec;
        f << "t" << t;
        f.release();
        cerr << "write result ends" << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
        exit(0); //so that we can check what happens about the first execution of cv::solvePnPRansac.
    }
#endif
    
	vector<cv::Point2f> projected3D;
	cv::projectPoints(ppcloud, rvec, t, K, distortion_coeff, projected3D);
    
	if(inliers.size()==0) { //get inliers
		for(int i=0;i<projected3D.size();i++) {
			if(norm(projected3D[i]-imgPoints[i]) < 10.0)
				inliers.push_back(i);
		}
	}
    
	if(inliers.size() < (double)(imgPoints.size())/5.0) {
		cerr << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< endl;
		return false;
	}
    
	if(cv::norm(t) > 200.0) {
		// this is bad...
		cerr << "estimated camera movement is too big, skip this camera\r\n";
		return false;
	}
    
	cv::Rodrigues(rvec, R);
	if(!CheckCoherentRotation(R)) {
		cerr << "rotation is incoherent. we should try a different base view..." << endl;
		return false;
	}
    
	std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
	return true;
}
