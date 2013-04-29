//
//  global.h
//  photo_tourism_core_code
//
//  Created by Zhang Yimeng on 13-4-28.
//  Copyright (c) 2013年 Zhang Yimeng. All rights reserved.
//

#ifndef photo_tourism_core_code_global_h
#define photo_tourism_core_code_global_h

// this is the flag for outputting additional debug info
#define PHOTO_TOURISM_DEBUG

// some constants for debugging
const char * const imageReadingDebugOutput = "imageReadingDebugOutput.xml";
const char * const descriptorDebugOutput = "descriptorDebugOutput.xml";
const char * const initialMatchMatrixDebugOutput = "initialMatchMatrixDebugOutput.xml";
const char * const refinedMatchMatrixDebugOutput = "refinedMatchMatrixDebugOutput.xml";

const char * const max2DDebugOutput = "max2DDebugOutput.xml";
const char * const max3DDebugOutput = "max3DDebugOutput.xml";
const char * const otherValueDebugOutput = "otherValueDebugOutput.xml";
const char * const resultDebugOutput = "resultDebugOutput.xml";

const char * const refinedMatchMatrixAfterTriangulationDebugOutput = "refinedMatchMatrixAfterTriangulationDebugOutput.xml";

const char * const baselineTriangulationDebugOutput = "baselineTriangulationDebugOutput.xml";
const char * const baselineTriangulationAfterBADebugOutput = "baselineTriangulationAfterBADebugOutput.xml";
const char * const finalProjectionMatrixOutput = "finalProjectionMatrixOutput.xml";
const char * const imageNameOutput = "imageNameOutput.xml";


#ifdef USE_PROFILING
#define CV_PROFILE(msg,code)	{\
std::cout << msg << " ";\
double __time_in_ticks = (double)cv::getTickCount();\
{ code }\
std::cout << "DONE " << ((double)cv::getTickCount() - __time_in_ticks)/cv::getTickFrequency() << "s" << std::endl;\
}
#else
#define CV_PROFILE(msg,code) code
#endif


#endif
