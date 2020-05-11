package app;

import java.util.Arrays;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.highgui.HighGui;

class testEntropy {

    public void run() {
        String filename = "C:\\Users\\RKT\\frc\\FRC2020\\Code\\Similar\\data\\lenabig.jpg";
        //filename = "blue.jpg";
    Mat src = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
    if (src.empty()) {
        System.out.println("Error opening image 1");
        System.exit(-1);
    }
    Mat graySrc = Mat.zeros(0,0,CvType.CV_8UC1);
    desaturate(src, graySrc);

    Mat grayHist = new Mat();

    // Compute histogram
    Imgproc.calcHist(Arrays.asList(graySrc), //histogram of 1 image only
            new MatOfInt(0), // the channel used
            new Mat(), // no mask is used
            grayHist, // the resulting histogram
            new MatOfInt(256), // number of bins, hist size
            new MatOfFloat(0.0f, 255.0f) // BRG range
    );
 
    int[] hist = new int[256];

    for (int idx = 0; idx < grayHist.rows(); ++idx) {
        hist[idx] = (int)grayHist.get(idx, 0)[0];
    }

    int entropySplit = Entropy_Threshold.entropySplit(hist);
    Mat otsu = new Mat();
    double otsuSplit = Imgproc.threshold(graySrc, otsu, 0., 255., Imgproc.THRESH_OTSU);
    Imgproc.threshold(graySrc, graySrc, (double)entropySplit, 255., Imgproc.THRESH_BINARY);
    
    System.err.println("otsu threshold " + otsuSplit);
    System.err.println("entropy threshold " + entropySplit);

    // normalize bin quantity
    // find maximum number in a bin
    double maxBin = -1.;
    for (int idx = 0; idx < 256; idx++) {
        if(maxBin < (int)grayHist.get(idx, 0)[0]) maxBin = grayHist.get(idx, 0)[0];
    }
    //System.err.println(maxBin);
    // normalize bin to 270 arbitrary - change to adjust to num rows of drawing mat
    for (int idx = 0; idx < 256; idx++) {
    grayHist.put(idx, 0, grayHist.get(idx, 0)[0]*270./maxBin);
    }

    Mat histDrawing = Mat.zeros(280, 280, CvType.CV_8UC1);

    for(int idx = 0; idx < hist.length; idx++)
        Imgproc.line(histDrawing, new Point(idx+5, 275), new Point(idx+5, 275-(int)grayHist.get(idx,0)[0]), new Scalar(255,0,0,0));

    Imgproc.line(histDrawing, new Point(entropySplit+5, 280), new Point(entropySplit+5, 0),  new Scalar(255));

    Mat diff = new Mat();
    Core.bitwise_xor(graySrc, otsu, diff);
    HighGui.imshow("histogram", histDrawing);
    HighGui.imshow("entropy", graySrc);
    HighGui.imshow("otsu", otsu);
    HighGui.imshow("diff", diff);
    HighGui.waitKey(0);
    System.exit(0);
    }
/**
	 * Converts a color image into shades of grey.
	 * @param input The image on which to perform the desaturate.
	 * @param output The image in which to store the output.
	 */
	static private void desaturate(Mat input, Mat output) {
		switch (input.channels()) {
			case 1:
				// If the input is already one channel, it's already desaturated
				input.copyTo(output);
				break;
			case 3:
				Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2GRAY);
				break;
			case 4:
				Imgproc.cvtColor(input, output, Imgproc.COLOR_BGRA2GRAY);
				break;
			default:
				throw new IllegalArgumentException("Input to desaturate must have 1, 3, or 4 channels");
		}
	}
}