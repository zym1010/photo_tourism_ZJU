//
//  main.cpp
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#include <iostream>
#include "global.h"

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
    
    
    std::cout << "Hello, World!\n";
    return 0;
}

