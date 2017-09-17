#pragma once
//utility.hpp

#include "opencv2/highgui/highgui.hpp"
using namespace std;

namespace utility
{
    double dist(CvPoint a, CvPoint b);
    double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center);
    string type2str(int type);
}