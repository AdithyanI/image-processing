#include "stdafx.h"

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/photo.hpp>
#include "external_tools/cartoon/cartoon.h"            // Cartoonify a photo.
#include "external_tools/cartoon/ImageUtils.h" 

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


// First class to have basic functions. 
// All basic functions such as reading, storing original image, displaying images will be here.
class basic {

protected:  // Keep it protected so that only me and my children can access it
    Mat image;  // The original image

public:

    basic(Mat src)  // constructor for basic class
    {
        if (!src.data)
            cout << "Consrtuctor invoke with no original image to be set." << endl;
        else {
            image = src.clone();
            cout << "Image successfully set." << endl;
        }
    }

    void showimage(std::string window_name)  //display stored image using a class function
    {
        if (!image.data)
            cout << "No data in " + window_name + " to show." << std::endl;
        else {
            namedWindow(window_name, WINDOW_AUTOSIZE);
            imshow(window_name, image);
        }
    }

};


// Creating a class where I need to manipulate by filtering the images. 
// Publicly inherits from the basic function.
class filter : public basic {

public:
    Mat filtered_image;     // The filtered image will be stored into this

public:

    filter(Mat src) : basic(src) // constructor for the filter class, also need to invoke basic class constructor with required argument
    { 
        filtered_image = image.clone();
    }

    void negative(Mat src)  // converting the image to negative
    {
        src = Scalar(255, 255, 255) - src;
    }

    void grayscale(Mat src) // convert the image into grayscale
    {
        // There are many ways to get a grayscale. Here we essentially convert a RGB to YCbCr,
        // and extract only the Y component. cvtColor is an inbuilt library. 
        // The exact same thing can be achieved by using kernel and transform too.
        // Since, anyway I am demonstrating that in the third method, I am skipping it here.
        // Convert roi_2 (3d) to roiGray (1d) corresponding grayscale image. 
        // Convert roiGray (1d) to roiGray_3d (3d) corresponding grayscale image in BGR format. 
        Mat tmp_gray_1d;
        Mat tmp_gray_3d;
        cvtColor(src, tmp_gray_1d, COLOR_BGR2GRAY);
        cvtColor(tmp_gray_1d, tmp_gray_3d, COLOR_GRAY2BGR);
        tmp_gray_3d.copyTo(src);
    }

    void blur(Mat src) // blur the given image
    {
        // Convert into gaussian blue. Using the gaussian blur inbuilt function. 
        // The variance and mean of blur across individual axis can be controlled individually.
        GaussianBlur(src, src, Size(), 5);
    }

    void sepia(Mat src) // apply sepia filter to the image
    {
        // Convert into sepia.
        // We can play around with kernel values to get the desired filter we need.
        // The following kernel values are for Sepia.
        /*
        outputBlue  = (inputRed * .272) + (inputGreen *.534) + (inputBlue * .131)
        outputGreen = (inputRed * .349) + (inputGreen *.686) + (inputBlue * .168)
        outputRed   = (inputRed * .393) + (inputGreen *.769) + (inputBlue * .189)
        */
        cv::Mat kernel_sepia = (cv::Mat_<float>(3, 3)
            <<
            0.272, 0.534, 0.131,
            0.349, 0.686, 0.168,
            0.393, 0.769, 0.189);
        transform(src, src, kernel_sepia);
    }

    void verticalStrip(Mat src) // 4 vertical strips of filter
    {
        negative(src(Rect(0, 0, src.cols / 4, src.rows)));
        grayscale(src(Rect(src.cols / 4, 0, src.cols / 4, src.rows)));
        blur(src(Rect(2 * src.cols / 4, 0, src.cols / 4, src.rows)));
        sepia(src(Rect(3 * src.cols / 4, 0, src.cols / 4, src.rows)));
    }

    void cartoonify_1(Mat src) {
        // Based on the tutorial here : http://www.learnopencv.com/non-photorealistic-rendering-using-opencv-python-c/
        //detailEnhance(src, src, 140, 61);
        stylization(src, src, 20, 0.7f);
        edgePreservingFilter(src, src, 1, 140, 0.1f);
    }

    void cartoonify_2(Mat src) {
        // Based on the tutorial here
        edgePreservingFilter(src, src, 1, 10, 0.1f);
        Mat img_gray;
        Mat img_blur;
        Mat img_edge;
        cvtColor(src, img_gray, COLOR_BGR2GRAY);
        medianBlur(img_gray, img_blur, 3);
        adaptiveThreshold(img_blur, img_edge, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 2);
        cvtColor(img_edge, img_edge, COLOR_GRAY2RGB);
        bitwise_and(src, img_edge, src);
        Mat dummy;
    }

    void cartoonify_3(Mat src) {
        // Based on the tutorial here : http://www.learnopencv.com/non-photorealistic-rendering-using-opencv-python-c/
        detailEnhance(src, src, 140, 0.1f);
        edgePreservingFilter(src, src, 1, 140, 0.1f);
        //stylization(src, src, 20, 0.7);
        //edgePreservingFilter(src, src, 1, 140, 0.1);
    }

    void applyFilter(int desired_filter) // apply the desired filter according to the option chosen
    {
        switch (desired_filter)
        {
        case(0):
            negative(filtered_image);
            break;
        case(1):
            grayscale(filtered_image);
            break;
        case(2):
            blur(filtered_image);
            break;
        case(3):
            sepia(filtered_image);
            break;
        case(4):
            verticalStrip(filtered_image);
            break;
        case(5):
            cartoonify_1(filtered_image);
            break;
        case(6):
            cartoonify_2(filtered_image);
            break;
        case(7):
            cartoonify_3(filtered_image);
            break;
        default:
            break;
        }
    }

    void compareDisplay() // display both images side by side
    {
        //showimage("Original image");
        if (!filtered_image.data)
            cout << "Filtered image data is empty. Nothing to show." << std::endl;
        else {
            namedWindow("Original image", CV_WINDOW_NORMAL);
            imshow("Original image", image);
            namedWindow("Filtered image", CV_WINDOW_NORMAL);
            imshow("Filtered image", filtered_image);
        }
    }

    void write(std::string file_name)
    {
        Mat newImage;
        hconcat(image, filtered_image, newImage);
        imwrite("images/" + file_name + ".jpg", newImage);
    }


};

// main() starts here
int main(int argc, char** argv)
{
    /*
    if (argc != 2) {
    fprintf(stderr, "Error! Correct usage: %s image_file_name \n", argv[0]);
    return 1;
    }

    printf("Loading input image.\n");
    Mat img = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    imwrite("Original_image.png", img);

    */

    // ONLY FOR TESTING - Standalone testing
    // read the image data in the file "MyPic.JPG" and store it in 'img'
    // CV_LOAD_IMAGE_UNCHANGED - image-depth=8 bits per pixel in each channel,  no. of channels=unchanged 
    // various possible options exist for this

    Mat img = imread("images/sample.jpeg", CV_LOAD_IMAGE_UNCHANGED);


    if (img.empty()) //check whether the image is loaded or not
    {
        cout << "Error : Image cannot be loaded..!!" << endl;
        system("pause"); //wait for a key press
        return -1;
    }

    if (!img.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }


    filter negative(img);
    negative.applyFilter(0);
    negative.write("negative");

    filter grayscale(img);
    grayscale.applyFilter(1);
    grayscale.write("grayscale");

    filter blur(img);
    blur.applyFilter(2);
    blur.write("blur");

    filter sepia(img);
    sepia.applyFilter(3);
    sepia.write("sepia");

    filter vertical(img);
    vertical.applyFilter(4);
    vertical.write("vertical");
    
    filter cartoon_image_1(img);
    cartoon_image_1.applyFilter(5);
    cartoon_image_1.write("cartoon_1");

    filter cartoon_image_2(img);
    cartoon_image_2.applyFilter(6);
    cartoon_image_2.write("cartoon_2");

    filter cartoon_image_3(img);
    cartoon_image_3.applyFilter(7);
    cartoon_image_3.write("cartoon_3");
    
    waitKey(0);
    return 0;
}

