//
//  FindCameraMatrices.cpp
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#include "FindCameraMatrices.h"
#include <cassert>
#include "Triangulation.h"
#include "global.h"
using namespace std;
using namespace cv;

const static double frontPercentageThreshold = 0.75;
const static double reprojectionErrorThreshold = 250.0;

void static TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
	//Using OpenCV's SVD
	SVD svd(E,SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
#ifdef PHOTO_TOURISM_DEBUG
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "----------------------- SVD ------------------------\n";
        cerr << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
        cerr << "----------------------------------------------------\n";
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
#endif
}



bool static DecomposeEtoRandT(
                              Mat_<double>& E,
                              Mat_<double>& R1,
                              Mat_<double>& R2,
                              Mat_<double>& t1,
                              Mat_<double>& t2)
{
    
	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);
    
	Matx33d W(0,-1,0,	//HZ 9.13
              1,0,0,
              0,0,1);
	Matx33d Wt(0,1,0,
               -1,0,0,
               0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
    
    
	return true;
}

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
    {
        cerr << "======DEBUGGING INFO BEGINS======" << endl;
        cerr << "F keeping " << cv::countNonZero(status) << " / " << status.size() << endl;
        cerr << "======DEBUGGING INFO ENDS======" << endl;
    }
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


bool FindCameraMatrices(const Mat& K,
						const Mat& Kinv,
						const Mat& distcoeff,
						const vector<KeyPoint>& imgpts1,
						const vector<KeyPoint>& imgpts2,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						Matx34d& P,
						Matx34d& P1,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud
						)
{
	//Find camera matrices
    
    //ZYM: repetition again........
    //ZYM: now we get the F between these two images.
    //ZYM: P2 * F * P1 = 0.
    Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches);
    //ZYM: now the matches are being used. In fact, I think here's a bug, since matches will be changed.
    //ZYM: check if we have enough number of matches.
    if(matches.size() < 100) { // || ((double)imgpts1_good.size() / (double)imgpts1.size()) < 0.25
        cerr << "not enough inliers after F matrix" << endl;
        return false;
    }
    
    
    
    //Essential matrix: compute then extract cameras [R|t]
    Mat_<double> E = K.t() * F * K; //according to HZ (9.12)
    
    //according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
    //ZYM: I think here R_ and t_ corresponds to all possible R's and t's.
    Mat_<double> R1(3,3);
    Mat_<double> R2(3,3);
    Mat_<double> t1(1,3);
    Mat_<double> t2(1,3);
    
    
    //        Mat_<double> rotation_vector(3,1);
    
    //decompose E to P' , HZ (9.19)
    //ZYM: This is the most useful part.
    {
        
        //ZYM: what we modify is P1.
        if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;
        
        
        //ZYM: flip the sign of E.
        if(determinant(R1)+1.0 < 1e-09) {
            //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
            //				cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
            E = -E;
            DecomposeEtoRandT(E,R1,R2,t1,t2);
        }
        
        
        //ZYM: first possibility of P1, using R1 and t1.
        P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
                     R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
                     R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
        //			cout << "Testing P1 " << endl << Mat(P1) << endl;
        //            Rodrigues(R1, rotation_vector);
        //            cout << "rotation vector " << endl << rotation_vector << endl;
        
        
        vector<CloudPoint> pcloud,pcloud1; vector<KeyPoint> corresp;
        double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
        double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
        vector<uchar> tmp_status;
        //check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
        if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > reprojectionErrorThreshold || reproj_error2 > reprojectionErrorThreshold) {
            
            //ZYM: second possibility of P1, using R1 and t2.
            P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
                         R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
                         R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
            
            pcloud.clear(); pcloud1.clear(); corresp.clear();
            reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
            reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
            
            if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > reprojectionErrorThreshold || reproj_error2 > reprojectionErrorThreshold) {
                
                
                //ZYM: third possibility of P1, using R2 and t1.
                P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
                             R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
                             R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
                
                pcloud.clear(); pcloud1.clear(); corresp.clear();
                reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
                reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
                
                if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > reprojectionErrorThreshold || reproj_error2 > reprojectionErrorThreshold) {
                    
                    //ZYM: fourth possibility of P1, using R2 and t2.
                    P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
                                 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
                                 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
                    
                    pcloud.clear(); pcloud1.clear(); corresp.clear();
                    reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
                    reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
                    
                    if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > reprojectionErrorThreshold || reproj_error2 > reprojectionErrorThreshold) {
                        //							cout << "Shit." << endl;
                        return false;
                    }
                }
            }
        }
        for (unsigned int i=0; i<pcloud.size(); i++) {
            outCloud.push_back(pcloud[i]);
        }
    }
	
	return true;
}



bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status) {
	vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	Matx44d P4x4 = Matx44d::eye();
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];
	
	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);
    
	double percentage = ((double)count / (double)pcloud.size());
    cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < frontPercentageThreshold){
		return false; //less than frontPercentageThreshold*100 % of the points are in front of the camera
    }
    
    
	return true;
}


bool CheckCoherentRotation(cv::Mat_<double>& R) {
    
	if(fabs(determinant(R))-1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}
    
	return true;
}
