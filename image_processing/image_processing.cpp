#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// First class to have basic functions. 
// All basic functions such as reading, storing original image, displaying images will be here.
class basic {

protected:  // Keep it protected so that only me and my children can access it
    Mat image;  // The original image

public:
    void setimage(Mat src)  //storing image into a class variable named image
    {
        if (!src.data)
            cout << "Error : The function is called with an empty argument." << endl;
        else {
            image = src.clone();
            cout << "Image successfully copied." << endl;
        }
    }

    void showimage(std::string window_name)    //display stored image using a class function
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

protected:
    Mat filtered_image;     // The filtered image will be stored into this

public:

    void negative()  // converting the image to negative
    {
        filtered_image = Scalar(255, 255, 255) - image;
    }

    void grayscale() // convert the image into grayscale
    {
        // There are many ways to get a grayscale. Here we essentially convert a RGB to YCbCr,
        // and extract only the Y component. cvtColor is an inbuilt library. 
        // The exact same thing can be achieved by using kernel and transform too.
        // Since, anyway I am demonstrating that in the third method, I am skipping it here.
        cvtColor(image, filtered_image, COLOR_BGR2GRAY);
    }

    void blur() // blur the given image
    {
        // Convert into gaussian blue. Using the gaussian blur inbuilt function. 
        // The variance and mean of blur across individual axis can be controlled individually.
        GaussianBlur(image, filtered_image, Size(), 5);
    }

    void sepia() // apply sepia filter to the image
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
        transform(image, filtered_image, kernel_sepia);
    }

    void applyFilter(int desired_filter) // apply the desired filter according to the option chosen
    {
        switch (desired_filter)
        {
        case(0):
            negative();
            break;
        case(1):
            grayscale();
            break;
        case(2):
            blur();
            break;
        case(3):
            sepia();
            break;
        default:
            break;
        }
    }


    void compareDisplay() // display both images side by side
    {
        showimage("Original image");
        if (!filtered_image.data)
            cout << "Filtered image data is empty. Nothing to show." << std::endl;
        else {
            namedWindow("Filtered image", WINDOW_AUTOSIZE);
            imshow("Filtered image", filtered_image);
        }
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
    Mat img = imread("images/lena.tif", CV_LOAD_IMAGE_UNCHANGED);


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

    // create a object with filter class
    // call the corresponding function
    filter negative_image;
    negative_image.setimage(img);
    negative_image.applyFilter(3);
    negative_image.compareDisplay();
    waitKey(0);
    return 0;
}