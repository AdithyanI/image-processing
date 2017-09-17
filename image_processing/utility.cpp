#include "utility.h"
#include "opencv2/highgui/highgui.hpp"


namespace utility
{
    // Helper function to calculate the distance between 2 points.
    double dist(CvPoint a, CvPoint b)
    {
        return sqrt(pow((double)(a.x - b.x), 2) + pow((double)(a.y - b.y), 2));
    }

    // Helper function that computes the longest distance from the edge to the center point.
    double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center)
    {
        // given a rect and a line
        // get which corner of rect is farthest from the line

        std::vector<cv::Point> corners(4);
        corners[0] = cv::Point(0, 0);
        corners[1] = cv::Point(imgSize.width, 0);
        corners[2] = cv::Point(0, imgSize.height);
        corners[3] = cv::Point(imgSize.width, imgSize.height);

        double maxDis = 0;
        for (int i = 0; i < 4; ++i)
        {
            double dis = dist(corners[i], center);
            if (maxDis < dis)
                maxDis = dis;
        }

        return maxDis;
    }

    string type2str(int type) {
        string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
        }

        r += "C";
        r += (chans + '0');

        return r;
    }

}