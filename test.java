package app;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

class SmoothingRun {

    /// Global Variables
    int DELAY_CAPTION = 1;// 500;
    int DELAY_BLUR = 1;// 00;
    int MAX_KERNEL_LENGTH = 31;
    Mat dst = new Mat();

    Mat src = new Mat();
    Mat srcA = new Mat();
    Mat srcB = new Mat();
    Mat dstU = new Mat(); // via LUT
    Mat dstV = new Mat();
    Mat dstU2 = new Mat(); // via double convert
    Mat dstV2 = new Mat();

    String windowName = "Filter Demo 1";

    public void xrun(String[] args) throws FileNotFoundException {
        
        Hist hist = new Hist();
        // create file for the System.out
		FileOutputStream fout=new FileOutputStream("out.txt");   
		PrintStream out=new PrintStream(fout); 
		System.setOut(out);
        //! create file for the System.out
        
        // String filename = ((args.length > 0) ? args[0] : "../data/lena.jpg");
        String filename1 = ((args.length > 0) ? args[0]
            : "cyan.jpg");
//        : "C:\\Users\\RKT\\frc\\FRC2020\\Code\\Similar\\data/lenabig.jpg");

        src = Imgcodecs.imread(filename1, Imgcodecs.IMREAD_COLOR);
        if (src.empty()) {
            System.out.println("Error opening image 1");
            System.out.println("Usage: ./Smoothing [image_name -- default ../data/lena.jpg] \n");
            System.exit(-1);
        }

        //if (true ) return;

        // LUT
            Mat lutU = new Mat(256, 1, CvType.CV_8UC3);
            Mat lutV = new Mat(256, 1, CvType.CV_8UC3);
            byte[] data = new byte[3];
            
            // for (short row = 0; row < 256; row++) {
            //     data[0] = (byte)row;
            //     System.err.println(data[0]);} //  0 to 255 is 0 to 127, -128 to -1

            for (short row = 0; row < 256; row++) {
            // U green to blue
            data[0] = (byte)(row);  // b
            data[1] = (byte)(255 - row); // g
            data[2] = (byte)(0); // r
            lutU.put(row, 0, data); // b g r

            // V green to red
            data[0] = (byte)(0);  // b
            data[1] = (byte)(255 - row); // g
            data[2] = (byte)(row); // r
            lutV.put(row, 0, data);  // b g r
            }
            // System.err.println(lut);
            // System.err.println(lut.dump());
        //! LUT

            List<Mat> channelsYUV = new ArrayList<Mat>();
            int normalizeType = Core.NORM_MINMAX;
            double normalizeAlpha = 0.0;
            double normalizeBeta = 255.0;
     
            Imgproc.cvtColor(src, dstU, Imgproc.COLOR_BGR2YUV);
            Imgproc.cvtColor(src, dstV, Imgproc.COLOR_BGR2YUV);

            // Core.multiply(dstA, new Scalar(0., 1., 0.), dstA);
            Core.split(dstU, channelsYUV); // y u v
            Imgproc.cvtColor(channelsYUV.get(1), dstU, Imgproc.COLOR_GRAY2BGR);
            Core.LUT(dstU, lutU, dstU);
            List<Mat> channelsU = new ArrayList<Mat>();
            Core.split(dstU, channelsU);
            Imgproc.equalizeHist(channelsU.get(0), channelsU.get(0));
            Imgproc.equalizeHist(channelsU.get(1), channelsU.get(1));
            Imgproc.equalizeHist(channelsU.get(2), channelsU.get(2));
            Core.merge(channelsU, dstU);
            //Core.add(dstU, new Scalar(-10., -10., -10.), dstU);
            //Core.normalize(dstU, dstU, normalizeAlpha, normalizeBeta, normalizeType);
            

            // Core.multiply(dstB, new Scalar(0., 0., 1.), dstB);
            Core.split(dstV, channelsYUV); // y u v
            Imgproc.cvtColor(channelsYUV.get(2), dstV, Imgproc.COLOR_GRAY2BGR);
            Core.LUT(dstV, lutV, dstV);
            //Core.add(dstV, new Scalar(-10., -10., -10.), dstV);
            //Core.normalize(dstV, dstV, normalizeAlpha, normalizeBeta, normalizeType);


            //Core.LUT(channelsYUV.get(0), lut, dstB);
            //Core.split(dstB, channelsYUV); // y u v

//         channelsYUVa.set(0, Mat.zeros(channelsYUVa.get(0).rows(), channelsYUVa.get(0).cols(), channelsYUVa.get(0).type()));
//         channelsYUVa.set(1, Mat.zeros(channelsYUVa.get(0).rows(), channelsYUVa.get(0).cols(), channelsYUVa.get(0).type()));
//   //      Core.add(channelsYUVa.get(0), new Scalar(50.), channelsYUVa.get(0));

//         channelsYUVb.set(0, Mat.zeros(channelsYUVb.get(0).rows(), channelsYUVb.get(0).cols(), channelsYUVb.get(0).type()));
//         channelsYUVb.set(2, Mat.zeros(channelsYUVb.get(0).rows(), channelsYUVb.get(0).cols(), channelsYUVb.get(0).type()));
//    //     Core.add(channelsYUVb.get(0), new Scalar(50.), channelsYUVb.get(0));

//         // HighGui.imshow("Y", channelsYUV.get(0));
//         // HighGui.imshow("U", channelsYUV.get(1));
//         // HighGui.imshow("V", channelsYUV.get(2));
  
//         Core.merge(channelsYUVa, dstA);
//         Core.merge(channelsYUVb, dstB);
        Imgproc.cvtColor(src, dstU2, Imgproc.COLOR_BGR2YUV);
        Imgproc.cvtColor(src, dstV2, Imgproc.COLOR_BGR2YUV);

        Core.multiply(dstU2, new Scalar(0., 1., 0.), dstU2);
        Imgproc.cvtColor(dstU2, dstU2, Imgproc.COLOR_YUV2BGR);
        Core.add(dstU2, new Scalar(65., 65., 0.), dstU2);
        Core.multiply(dstU2, new Scalar(1., 1., 0.), dstU2);
    
        Core.multiply(dstV2, new Scalar(0., 0., 1.), dstV2);
        Imgproc.cvtColor(dstV2, dstV2, Imgproc.COLOR_YUV2BGR);
        Core.add(dstV2, new Scalar(0., 65., 65.), dstV2);
        Core.multiply(dstV2, new Scalar(0., 1., 1.), dstV2);
 
        hist.displayHist(dstU);
        hist.displayHist(dstV);
        hist.displayHist(dstU2);
        hist.displayHist(dstV2);
        HighGui.imshow("U channel", dstU);
        HighGui.imshow("V channel", dstV);
        HighGui.imshow("U2 channel", dstU2);
        HighGui.imshow("V2 channel", dstV2);

        // HighGui.imshow("L", channelsLab.get(0));
        // HighGui.imshow("a", channelsLab.get(1));
        // HighGui.imshow("b", channelsLab.get(2));

        HighGui.waitKey(0);

        if(true) return;

        if( displayCaption( "Original Image" ) != 0 ) { System.exit(0); }

        dst = src.clone();
        if( displayDst( DELAY_CAPTION ) != 0 ) { System.exit(0); }

        /// Applying Homogeneous blur
        if( displayCaption( "Homogeneous Blur" ) != 0 ) { System.exit(0); }

        //! [blur]
        for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
            Imgproc.blur(src, dst, new Size(i, i), new Point(-1, -1));
            displayDst(DELAY_BLUR);
        }
        //! [blur]

        /// Applying Gaussian blur
        if( displayCaption( "Gaussian Blur" ) != 0 ) { System.exit(0); }

        //! [gaussianblur]
        for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
            Imgproc.GaussianBlur(src, dst, new Size(i, i), 0, 0);
            displayDst(DELAY_BLUR);
        }
        //! [gaussianblur]

        /// Applying Median blur
        if( displayCaption( "Median Blur" ) != 0 ) { System.exit(0); }

        //! [medianblur]
        for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
            Imgproc.medianBlur(src, dst, i);
            displayDst(DELAY_BLUR);
        }
        //! [medianblur]

        /// Applying Bilateral Filter
        if( displayCaption( "Bilateral Blur" ) != 0 ) { System.exit(0); }

        //![bilateralfilter]
        for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
            Imgproc.bilateralFilter(src, dst, i, i * 2, i / 2);
            displayDst(DELAY_BLUR);
        }
        //![bilateralfilter]

        /// Done
        displayCaption( "Done!" );
return;
//        System.exit(0);
    }

    int displayCaption(String caption) {
        dst = Mat.zeros(src.size(), src.type());
        Imgproc.putText(dst, caption,
                new Point(src.cols() / 4, src.rows() / 2),
                Imgproc.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 255, 255));

        return displayDst(DELAY_CAPTION);
    }

    int displayDst(int delay) {
        HighGui.imshow( windowName, dst );
        int c = HighGui.waitKey( 0/*delay*/ );
        if (c >= 0) { return -1; }
        return 0;
    }
//}

//public class Smoothing {
    public static void main(String[] args) throws FileNotFoundException {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new testEntropy().run();
        
        if(true) return;
        new SmoothingRun().xrun(args);
 
        System.exit(0);
    }
}

// Mat dst = Mat.zeros(2, 21, CvType.CV_8UC1);
//         Core.scaleAdd(Mat.ones(2, 2, CvType.CV_8UC1), 8.4, Mat.eye(2, 2, CvType.CV_8UC1), dst);
//         System.err.println(dst.dump());
//         Core.add(Mat.ones(2, 2, CvType.CV_8UC1), new Scalar(8.4), dst);
//         System.err.println(dst.dump());
//         if(true) return;


// mRotation[0] = (float)((rawData[2] << 8) | (rawData[3] & 0xFF)); // x

/*
import cv2
import numpy as np


def make_lut_u():  // green to blue
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8) // b g r

def make_lut_v():  // green to red
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8) // b g r


img = cv2.imread('shed.png')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(img_yuv)

lut_u, lut_v = make_lut_u(), make_lut_v()

# Convert back to BGR so we can apply the LUT and stack the images
y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

u_mapped = cv2.LUT(u, lut_u)
v_mapped = cv2.LUT(v, lut_v)

result = np.vstack([img, y, u_mapped, v_mapped])

cv2.imwrite('shed_combo.png', result)

*/


// Mat testit = new Mat();
// testit = Mat.eye(3, 3, CvType.CV_8UC1);
// Core.add(testit, new Scalar(255.), testit);
// System.err.println(testit.dump());
// Core.multiply(testit, new Scalar(300.), testit);
// System.err.println(testit.dump());

 

/*
@Override
public void process(Mat input, Mat mask) {
    channels = new ArrayList<>();

    switch(color){
        case RED:
            if(threshold == -1){
                threshold = 164;
            }

            Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2Lab);
            Imgproc.GaussianBlur(input,input,new Size(3,3),0);
            Core.split(input, channels);
            Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY);
            break;
        case BLUE:
            if(threshold == -1){
                threshold = 145;
            }

            Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2YUV);
            Imgproc.GaussianBlur(input,input,new Size(3,3),0);
            Core.split(input, channels);
            Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY);
            break;
        case YELLOW:
            if(threshold == -1){
                threshold = 95;
            }

            Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2YUV);
            Imgproc.GaussianBlur(input,input,new Size(3,3),0);
            Core.split(input, channels);
            Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY_INV);
            break;
    }

    for(int i=0;i<channels.size();i++){
        channels.get(i).release();
    }

    input.release();

}



//////////////////////////


package com.disnodeteam.dogecv.filters;

import android.graphics.Color;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/ **
 * Created by Victo on 1/1/2018.
 * /

public class LeviColorFilter extends DogeCVColorFilter {
    public enum ColorPreset{
        RED,
        BLUE,
        YELLOW
    }
    private ColorPreset color = ColorPreset.RED;
    private double threshold = -1; // if -1 the color mode will use its own defaults
    private List<Mat> channels = new ArrayList<>();

    public LeviColorFilter(ColorPreset filterColor){
        color = filterColor;
    }

    public LeviColorFilter(ColorPreset filterColor, double filterThreshold){
        color = filterColor;
        filterThreshold = filterThreshold;
    }



    @Override
    public void process(Mat input, Mat mask) {
        channels = new ArrayList<>();

        switch(color){
            case RED:
                if(threshold == -1){
                    threshold = 164;
                }

                Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2Lab);
                Imgproc.GaussianBlur(input,input,new Size(3,3),0);
                Core.split(input, channels);
                Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY);
                break;
            case BLUE:
                if(threshold == -1){
                    threshold = 145;
                }

                Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2YUV);
                Imgproc.GaussianBlur(input,input,new Size(3,3),0);
                Core.split(input, channels);
                Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY);
                break;
            case YELLOW:
                if(threshold == -1){
                    threshold = 95;
                }

                Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2YUV);
                Imgproc.GaussianBlur(input,input,new Size(3,3),0);
                Core.split(input, channels);
                Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY_INV);
                break;
        }

        for(int i=0;i<channels.size();i++){
            channels.get(i).release();
        }

        input.release();

    }

    // RED FILTER
    public void leviRedFilter (Mat input, Mat mask){




    }

    public void leviRedFilter (Mat input, Mat mask, double threshold){


        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2Lab);
        Imgproc.GaussianBlur(input,input,new Size(3,3),0);
        Core.split(input, channels);
        Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY);

        for(int i=0;i<channels.size();i++){
            channels.get(i).release();
        }
    }


    // BLUE FILTER

    public void leviBlueFilter (Mat input, Mat mask){
        List<Mat> channels = new ArrayList<>();

        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2Lab);
        Imgproc.GaussianBlur(input,input,new Size(3,3),0);
        Core.split(input, channels);
        Imgproc.threshold(channels.get(1), mask, 145, 255, Imgproc.THRESH_BINARY);

        for(int i=0;i<channels.size();i++){
            channels.get(i).release();
        }
    }

    public void leviBlueFilter (Mat input, Mat mask, double threshold){
        List<Mat> channels = new ArrayList<>();

        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2YUV);
        Imgproc.GaussianBlur(input,input,new Size(3,3),0);
        Core.split(input, channels);
        Imgproc.threshold(channels.get(1), mask, threshold, 255, Imgproc.THRESH_BINARY);

        for(int i=0;i<channels.size();i++){
            channels.get(i).release();
        }
    }
}
*/

// List<Mat> channelsYUV = new ArrayList<Mat>();
            
// Imgproc.cvtColor(src, dstU, Imgproc.COLOR_BGR2YUV);
// Imgproc.cvtColor(src, dstV, Imgproc.COLOR_BGR2YUV);

// // Core.multiply(dstA, new Scalar(0., 1., 0.), dstA);
// Core.split(dstU, channelsYUV); // y u v
// Imgproc.cvtColor(channelsYUV.get(1), dstU, Imgproc.COLOR_GRAY2BGR);
// Core.LUT(dstU, lutU, dstU);

// // Core.multiply(dstB, new Scalar(0., 0., 1.), dstB);
// Core.split(dstV, channelsYUV); // y u v
// Imgproc.cvtColor(channelsYUV.get(2), dstV, Imgproc.COLOR_GRAY2BGR);
// Core.LUT(dstV, lutV, dstV);

// //Core.LUT(channelsYUV.get(0), lut, dstB);
// //Core.split(dstB, channelsYUV); // y u v

// //         channelsYUVa.set(0, Mat.zeros(channelsYUVa.get(0).rows(), channelsYUVa.get(0).cols(), channelsYUVa.get(0).type()));
// //         channelsYUVa.set(1, Mat.zeros(channelsYUVa.get(0).rows(), channelsYUVa.get(0).cols(), channelsYUVa.get(0).type()));
// //   //      Core.add(channelsYUVa.get(0), new Scalar(50.), channelsYUVa.get(0));

// //         channelsYUVb.set(0, Mat.zeros(channelsYUVb.get(0).rows(), channelsYUVb.get(0).cols(), channelsYUVb.get(0).type()));
// //         channelsYUVb.set(2, Mat.zeros(channelsYUVb.get(0).rows(), channelsYUVb.get(0).cols(), channelsYUVb.get(0).type()));
// //    //     Core.add(channelsYUVb.get(0), new Scalar(50.), channelsYUVb.get(0));

// //         // HighGui.imshow("Y", channelsYUV.get(0));
// //         // HighGui.imshow("U", channelsYUV.get(1));
// //         // HighGui.imshow("V", channelsYUV.get(2));

// //         Core.merge(channelsYUVa, dstA);
// //         Core.merge(channelsYUVb, dstB);
// Imgproc.cvtColor(src, dstU2, Imgproc.COLOR_BGR2YUV);
// Imgproc.cvtColor(src, dstV2, Imgproc.COLOR_BGR2YUV);

// Core.multiply(dstU2, new Scalar(0., 1., 0.), dstU2);
// Imgproc.cvtColor(dstU2, dstU2, Imgproc.COLOR_YUV2BGR);

// Core.multiply(dstV2, new Scalar(0., 0., 1.), dstV2);
// Imgproc.cvtColor(dstV2, dstV2, Imgproc.COLOR_YUV2BGR);
// //Core.add(channelsYUVa.get(0), new Scalar(50.), channelsYUVa.get(0));

// //Core.add(dstA, new Scalar(50., 50., 50.), dstA);
// hist.displayHist(dstU);
// //Imgproc.cvtColor(dstV, dstV, Imgproc.COLOR_YUV2BGR);
// //Core.add(dstB, new Scalar(50., 50., 50.), dstB);
// Mat correlateUB = Mat.zeros(256, 256, CvType.CV_8UC1);
// Mat correlateUG = Mat.zeros(256, 256, CvType.CV_8UC1);
// Mat correlateUR = Mat.zeros(256, 256, CvType.CV_8UC1);
// Mat correlateVB = Mat.zeros(256, 256, CvType.CV_8UC1);
// Mat correlateVG = Mat.zeros(256, 256, CvType.CV_8UC1);
// Mat correlateVR = Mat.zeros(256, 256, CvType.CV_8UC1);
// byte[] aRowU = new byte[256*3];
// byte[] aRowU2 = new byte[256*3];
// byte[] aRowV = new byte[256*3];
// byte[] aRowV2 = new byte[256*3];
// byte[] pixelOn = {-1};
// for (int i=0; i < 256; i++) {
// dstU.get(i, 0, aRowU);
// dstU2.get(i, 0, aRowU2);
// dstV.get(i, 0, aRowV);
// dstV2.get(i, 0, aRowV2);
//     //for (int idx = 0; idx < 256; idx++) System.err.println(aRowU[idx] + " " + aRowU2[idx]);
// for (int j=0; j < 256*3; j+=3)
// {
// correlateUB.put(aRowU[j], aRowU2[j], pixelOn);
// correlateUG.put(aRowU[j+1], aRowU2[j+1], pixelOn);
// correlateUR.put(aRowU[j+2], aRowU2[j+2], pixelOn);

// correlateVB.put(aRowV[j], aRowV2[j], pixelOn);
// correlateVG.put(aRowV[j+1], aRowV2[j+1], pixelOn);
// correlateVR.put(aRowV[j+2], aRowV2[j+2], pixelOn);
// }
// }
// //System.err.println(correlate.dump());

// hist.displayHist(dstV);
// hist.displayHist(dstU2);
// hist.displayHist(dstV2);
// HighGui.imshow("U channel", dstU);
// HighGui.imshow("V channel", dstV);
// HighGui.imshow("U2 channel", dstU2);
// HighGui.imshow("V2 channel", dstV2);

// HighGui.imshow("correlateU-U2", correlateU);
// HighGui.imshow("correlateV-V2", correlateV);
// // HighGui.imshow("L", channelsLab.get(0));
// // HighGui.imshow("a", channelsLab.get(1));
// // HighGui.imshow("b", channelsLab.get(2));

// HighGui.waitKey(0);

