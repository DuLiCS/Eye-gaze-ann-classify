//
//  main.cpp
//  Classify
//
//  Created by 杜立 on 14-9-9.
//  Copyright (c) 2014年 杜立. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv/ml.h>

//#include<opencv/cv.hpp>
//#include "opencv/highgui.h"


const int DATANUMBER=719;
const int numberCharacters=112;
const int nlayers=100;
int comparemartix[DATANUMBER][2];
CvANN_MLP ann;
int dist[DATANUMBER][1];
using namespace std;
using namespace cv;


void train(Mat TrainData, Mat classes, int nlayers){
    Mat layers(1,3,CV_32SC1);
    layers.at<int>(0)= TrainData.cols;
    layers.at<int>(1)= nlayers;
    layers.at<int>(2)= numberCharacters;
    ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);
    
    //Prepare trainClases
    //Create a mat with n trained data by m classes
    Mat trainClasses;
    trainClasses.create( TrainData.rows, numberCharacters, CV_32FC1 );
    for( int i = 0; i <  trainClasses.rows; i++ )
    {
        for( int k = 0; k < trainClasses.cols; k++ )
        {
            //If class of data i is same than a k class
            if( k == classes.at<int>(i) )
                trainClasses.at<float>(i,k) = 1;
            else
                trainClasses.at<float>(i,k) = 0;
        }
    }
    Mat weights( 1, TrainData.rows, CV_32FC1, Scalar::all(1) );
    
    //Learn classifier
    ann.train( TrainData, trainClasses, weights );
}


int classify(Mat f){
    Mat output(1, numberCharacters, CV_32FC1);
    ann.predict(f, output);
    Point maxLoc;
    double maxVal;
    minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
    //We need know where in output is the max val, the x (cols) is the class.
    cout<<maxLoc.x<<endl;
    return maxLoc.x;
}


void testprogramme(){
    double testdata[1][16];
    double testpoint[DATANUMBER][2];
    int testpointclassify[DATANUMBER][1];
    double a,b;
    ifstream ftestpoint("/Users/duli/Programme/Eye gaze/Classify/Database/randomCurves2.groundtruth");
    ifstream ftestdata("/Users/duli/Programme/Eye gaze/Classify/Database/randomCurves2.test");
    for (int i=0; i<DATANUMBER; i++) {
        ftestpoint>>testpoint[i][0]>>testpoint[i][1];
        a=int(testpoint[i][0]/100);
        b=int(testpoint[i][1]/100);
        testpointclassify[i][0]=(a*8+b);
    }
    for (int k=0; k<DATANUMBER; k++) {
        for (int j=0; j<16;j++) {
        
        ftestdata>>testdata[0][j];
        }
        Mat testmartix(1,16,CV_64FC1,testdata);
        comparemartix[k][0]=classify(testmartix);
        comparemartix[k][1]=testpointclassify[k][0];
        int x=comparemartix[k][0];
        int y=comparemartix[k][1];
        int c=x/8;
        int d=x%8;
        int m=y/8;
        int n=y%8;
        dist[k][0]=sqrt((c-m)*(c-m)+(d-n)*(d-n));
    }
    

    
}


//void trainmartix(Mat TrainData, Mat classes, int nlayers)
//{
//    // Set up BPNetwork's parameters
//    CvANN_MLP_TrainParams params;
//    params.train_method=CvANN_MLP_TrainParams::BACKPROP;
//    params.bp_dw_scale=0.1;
//    params.bp_moment_scale=0.1;
//    Mat layerSizes=(Mat_<int>(1,3) << 299,nlayers,112);
//    Mat trainClasses;
//    trainClasses.create( TrainData.rows, numberCharacters, CV_32FC1 );
//    for( int i = 0; i <  trainClasses.rows; i++ )
//    {
//        for( int k = 0; k < trainClasses.cols; k++ )
//        {
//            //If class of data i is same than a k class
//            if( k == classes.at<int>(i) )
//                trainClasses.at<float>(i,k) = 1;
//            else
//                trainClasses.at<float>(i,k) = 0;
//        }
//    }
//
//    ann.create(layerSizes,CvANN_MLP::SIGMOID_SYM);//CvANN_MLP::SIGMOID_SYM
//    //CvANN_MLP::GAUSSIAN
//    //CvANN_MLP::IDENTITY
//    ann.train(TrainData, trainClasses, Mat(),Mat(), params);
//
//}



int main(int argc, const char * argv[])
{
//    ifstream fpoint("/Users/duli/Programme/Eye gaze/Classify/Database/circles.groundtruth");
//    ifstream fdata("/Users/duli/Programme/Eye gaze/Classify/Database/circles.test");
//    double pointdata[DATANUMBER][2];
//    int pointclassify[DATANUMBER][1];
//    double facedata[DATANUMBER][16];
//    int x,y;
//    
//    //for (int j=0; j<8; j++) {
//    //    fdata>>facedata[0][j][0]>>facedata[0][j][1];
//    //}
//    // insert code here...
//    
//
    Mat data;
    Mat classes;
    FileStorage fs1("/Users/duli/Programme/Eye gaze/Classify/COMPARE.xml",FileStorage::WRITE);
    FileStorage fs("/Users/duli/Programme/Eye gaze/Classify/data.xml",FileStorage::READ);
    fs["data"]>>data;
    fs["type"]>>classes;
//    for (int i=0; i<DATANUMBER; i++) {
//        fpoint>>pointdata[i][0]>>pointdata[i][1];
//        x=int(pointdata[i][0]/100);
//        y=int(pointdata[i][1]/100);
//        pointclassify[i][0]=(x*8+y);
//        //pointclassify[i][2]=((pointclassify[i][0])*7+pointclassify[i][1]);
//        for (int k=0; k<16; k++) {
//            fdata>>facedata[i][k];
//        }
//    }
//    Mat ANNTRAINdata(299,16,CV_64FC1,facedata);
//     Mat ANNTRAINclass(299,1,CV_32SC1,pointclassify);
//    fs<<"TrainData"<<ANNTRAINdata;
//    fs<<"classes"<<ANNTRAINclass;
    train(data,classes,nlayers);
    testprogramme();
    Mat comparemartixmat(DATANUMBER,2,CV_32SC1,comparemartix);
    fs1<<"COMPARERESULT"<<comparemartixmat;
    Mat distmat(DATANUMBER,1,CV_32SC1,dist);
    fs1<<"result"<<distmat;
    
    //fpoint>>pointdata[298][0]>>pointdata[298][1];
    //pointclassify[298]=(int(pointdata[298][0]/100))*7+int(pointdata[298][1]/100);
    return 0;
}

