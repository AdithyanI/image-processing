#include "stdafx.h"

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "utility.h"

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

    void write(std::string file_name)
    {
        imwrite("images/" + file_name + ".jpg", image);
    }


};


// Creating a class where I need to manipulate by filtering the images. 
// Publicly inherits from the basic function.
class filter : public basic {

private:
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

    // Overloaded function, that horizontally concantes both the original
    // and the filtered image. Remove later when you do not need this.
    void write(std::string file_name)
    {
        //Mat newImage;
        //hconcat(image, filtered_image, newImage);
        imwrite("images/" + file_name + ".jpg", filtered_image);
    }

    // Helper function that creates a gradient image.   
    // firstPt, radius and power, are variables that control the artistic effect of the filter.
    void generateGradient(Mat& mask, Point refPt, double radius, double power)
    {
        //Point refPt = Point(mask.size().width / 2, mask.size().height / 2);
        //double radius = 1.0;
        //double power = 0.1;
        double maxImageRad = radius * utility::getMaxDisFromCorners(mask.size(), refPt); 
        // The radius determines maximum boundary where we can have a value of 1.

        mask.setTo(Scalar(1));
        double temp_distance;

        for (int i = 0; i < mask.rows; i++)
        {
            for (int j = 0; j < mask.cols; j++)
            {
                temp_distance = utility::dist(refPt, Point(j, i)) / maxImageRad; 
                // The farthest away point gives a value of 1, the closest point a value of 0.
                temp_distance = min(temp_distance, 1.0); 
                temp_distance = pow(temp_distance, power);
                mask.at<uchar>(i, j) = unsigned int(temp_distance * 255);
            }
        }
    }

    void vignettify(Point refPt, double radius, double power) { // apply vignette effect for the filter

        Mat background = image.clone();
        Mat foreground = filtered_image.clone();

        foreground.convertTo(foreground, CV_32FC3);
        background.convertTo(background, CV_32FC3);

        Mat maskImg(filtered_image.size(), CV_8UC1);
        generateGradient(maskImg, refPt,  radius,  power);
        cvtColor(maskImg, maskImg, CV_GRAY2RGB);
        //imwrite("images/vignette/mask.png", maskImg);
        maskImg.convertTo(maskImg, CV_32FC3, 1.0/255);

        // Multiply the foreground with the alpha matte
        multiply(maskImg, foreground, foreground);
        //imwrite("images/vignette/foreground.png", foreground);
        
        // Multiply the background with ( 1 - alpha )
        multiply(Scalar::all(1.0)-maskImg, background, background);
        //imwrite("images/vignette/background.png", background);

        // Add the masked foreground and background.
        add(foreground, background, filtered_image);
        filtered_image.convertTo(filtered_image, CV_8UC3);

        //imwrite("images/vignette/vignette.png", filtered_image);

    }

    void dodge(Mat image, Mat mask, Mat dst, int scale) {
        // Implemented based on the tutorial here : http://www.askaswiss.com/2016/01/how-to-create-pencil-sketch-opencv-python.html
        // Helper function for edge sketch.
        divide(image, 255 - mask, dst, scale);
    }

    void burn(Mat image, Mat mask, Mat dst, int scale) {
        // Implemented based on the tutorial here : http://www.askaswiss.com/2016/01/how-to-create-pencil-sketch-opencv-python.html
        // Helper function for edge sketch.
        divide(255 - image, 255 - mask, dst, scale);
        dst = 255 - dst;
    }

    void edgeSketch() {
        // Implemented based on the tutorial here : http://www.askaswiss.com/2016/01/how-to-create-pencil-sketch-opencv-python.html
        cvtColor(filtered_image, filtered_image, COLOR_BGR2GRAY);
        Mat img_gray = filtered_image.clone();
        filtered_image = 255 - filtered_image;
        GaussianBlur(filtered_image, filtered_image, Size(101, 101), 0.0, 0.0);
        dodge(img_gray, filtered_image, filtered_image, 256);        
        Mat img_canvas = imread("canvas.jpg", CV_8UC1);
        resize(img_canvas, img_canvas, Size(filtered_image.size().width, filtered_image.size().height));
        imshow("canvas", img_canvas);
        Mat img_new;

        string ty = utility::type2str(img_canvas.type());
        printf("Image matrix: %s %dx%d \n", ty.c_str(), img_canvas.cols, img_canvas.rows);

        string ty1 = utility::type2str(filtered_image.type());
        printf("Image matrix: %s %dx%d \n", ty1.c_str(), filtered_image.cols, filtered_image.rows);

        //addWeighted(filtered_image, 0.5, img_canvas, 0.5, 0.0, filtered_image);
        multiply(filtered_image, img_canvas, filtered_image, 1.0 / 256);
    }

    void contour(int thresh = 100) {
        // http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
        /// Convert image to gray and blur it

        cvtColor(filtered_image, filtered_image, CV_BGR2GRAY);
        GaussianBlur(filtered_image, filtered_image, Size(3, 3), 0.0, 0.0);

        Mat canny_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        /// Detect edges using canny
        Canny(filtered_image, canny_output, thresh, thresh * 2, 3);
        /// Find contours
        findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        /// Draw contours
        Mat drawing(canny_output.size(), CV_8UC3);
        drawing = Scalar(255, 255, 255);
        //drawContours(drawing, contours, -1, color, 2, 8, hierarchy, 0, Point());
        drawContours(drawing, contours, -1, 0, 3, 8 );
        GaussianBlur(drawing, drawing, Size(3, 3), 0.0, 0.0);
        drawing.copyTo(filtered_image);
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

    Mat img = imread("images/sample_1.jpg", CV_LOAD_IMAGE_UNCHANGED);

    string ty = utility::type2str(img.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), img.cols, img.rows);

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

    /*
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


    filter cartoon_image_3(img);
    cartoon_image_3.applyFilter(7);
    cartoon_image_3.write("cartoon_3");
    */

    //filter cartoon_image_2(img);
    //cartoon_image_2.applyFilter(6);
    //cartoon_image_2.write("cartoon_2");


    /*
    for (int i = 1; i <= 3; i = i + 2) {
        for (int j = 1; j <= 3; j = j + 2) {
            filter vignette(img);
            vignette.applyFilter(1);
            Point refPt(img.size().width * i/4, img.size().height * j/4);
            double radius = 0.4;
            double power = 1.2;
            vignette.vignettify(refPt, radius, power);
            //vignette.write("vignette/location/vignette_center_" + to_string(radius) + "_" + to_string(power));
            vignette.write("vignette/location/vignette_center_" + to_string(i) + "_" + to_string(j));
        }
    }

    filter edgesketch(img);
    edgesketch.edgeSketch();
    edgesketch.write("edgesketch/canvas_multiply");
    edgesketch.compareDisplay();
    */
    for (int i = 70; i <= 130; i += 20){
        filter edgesketch(img);
        edgesketch.contour(i);
        edgesketch.write("edgesketch/contour_" + to_string(i));
    }
    waitKey(0);
    return 0;
}

