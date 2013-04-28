//
//  main.cpp
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#include <iostream>
#include "global.h"
#include "helper.h"
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "MultiCameraPnP.h"

using namespace std;

int main(int argc, const char * argv[])
{
    
    //read arguments begins.
    if (argc < 2) {
		cerr << "USAGE: " << argv[0] << " <path_to_images> [downscale factor = 1.0]" << endl;
		return 0;
	}
    
    double downscale_factor = 1.0;
    
    if(argc >= 3){
		downscale_factor = atof(argv[2]);
    }
    
#ifdef PHOTO_TOURISM_DEBUG
    cerr << "======DEBUGGING INFO BEGINS======" << endl;
    cerr << "path of the images: " << argv[1] << endl;
    cerr << "scale factor: " << downscale_factor << endl;
    cerr << "======DEBUGGING INFO ENDS======" << endl;
#endif
    //read argument ends.

    
    std::vector<cv::Mat> images; //saves scaled images (not necessarily 3 channel, 8bit Unsigned)
    std::vector<std::string> images_names; //saves file names (without dir)
    
    //read images begins.
    open_imgs_dir(argv[1],images,images_names,downscale_factor);
    
    //ZYM: in case the supplied folder contains no valid image.
	if(images.size() == 0) {
		cerr << "can't get image files" << endl;
		return 1;
	}
    
#ifdef PHOTO_TOURISM_DEBUG
    cerr << "======DEBUGGING INFO BEGINS======" << endl;
    cerr << "write images to file begins" << endl;
    cv::FileStorage f;
    f.open(imageReadingDebugOutput, cv::FileStorage::WRITE);
    
    for (unsigned i = 0; i < images.size(); i++) {
        stringstream ss;
        ss << "img" << i; //names must begin with a letter
        cerr << ss.str() << endl;
        f << ss.str() << images[i];
    }
    f.release();
    cerr << "write images to file ends" << endl;
    cerr << "======DEBUGGING INFO ENDS======" << endl;
#endif
    
    //read images ends.
    
    //build the MultiCameraPnP object begins
    cv::Ptr<MultiCameraPnP> distance = new MultiCameraPnP(images,images_names);
    //build the MultiCameraPnP object ends
    
    //recover depth begins
    distance->RecoverDepthFromImages();
    //recover image ends
    
    std::cout << "Hello, World!\n";
    return 0;
}

