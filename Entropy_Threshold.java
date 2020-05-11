package app;

/* from an ImageJ plug-in filter but without the ImageJ references*/

/*
 * DAVID
 * 
 * This file modified by David Foster to improve clarity.
 */

/**
 * Automatic thresholding technique based on the entopy of the histogram.
 *
 * This plugin does automatic thresholding based on the entopy of the histogram.
 * The method is very similar to Otsu's method.
 * Rather than maximising the inter-class variance, it maximises the inter-class entropy.
 * Entropy is a measure of the uncertainity of an event taking place.
 * You can calculate it as: S = -(sum)p*log2(p) so it is very straightforward to do using the histogram data.
 * (p) is the probability of a pixel greyscale value in the image, and (sum) is the greek capital sigma.
 * It is customary to use log in base 2.
 *
 * See: P.K. Sahoo, S. Soltani, K.C. Wong and, Y.C. Chen "A Survey of
 * Thresholding Techniques", Computer Vision, Graphics, and Image
 * Processing, Vol. 41, pp.233-260, 1988.
 *
 * @author Jarek Sacha
 */
public class Entropy_Threshold {

 /**
  * Calculate maximum entropy split of a histogram.
  *
  * @param hist histogram to be thresholded.
  *
  * @return index of the maximum entropy split.`
  */
 public static int entropySplit(int[] hist) {
   /*
    * DAVID
    * 
    * For all i,
    *     normalizedHist[i] = (double) hist[i]/sum(hist)
    */
   // Normalize histogram, that is makes the sum of all bins equal to 1.
   double sum = 0;
   for (int i = 0; i < hist.length; ++i) {
     sum += hist[i];
   }
   if (sum == 0) {
     // This should not normally happen, but...
     throw new IllegalArgumentException("Empty histogram: sum of all bins is zero.");
   }
   double[] normalizedHist = new double[hist.length];
   for (int i = 0; i < hist.length; i++) {
     normalizedHist[i] = hist[i] / sum;
   }

   /*
    * DAVID
    * 
    * pT = cumulative_sum(normalizedHist)
    */
   double[] pT = new double[hist.length];
   pT[0] = normalizedHist[0];
   for (int i = 1; i < hist.length; i++) {
     pT[i] = pT[i - 1] + normalizedHist[i];
   }

   // Entropy for black and white parts of the histogram
   final double epsilon = Double.MIN_VALUE;
   double[] hB = new double[hist.length];
   double[] hW = new double[hist.length];
   for (int t = 0; t < hist.length; t++) {
     // Black entropy
     double pTB = pT[t];		// DAVID
     if (pTB > epsilon) {
       double hhB = 0;
       for (int i = 0; i <= t; i++) {
         if (normalizedHist[i] > epsilon) {
           hhB -= normalizedHist[i] / pTB * Math.log(normalizedHist[i] / pTB);
         }
       }
       hB[t] = hhB;
     } else {
       hB[t] = 0;
     }

     // White  entropy
     double pTW = 1 - pT[t];
     if (pTW > epsilon) {
       double hhW = 0;
       for (int i = t + 1; i < hist.length; ++i) {
         if (normalizedHist[i] > epsilon) {
           hhW -= normalizedHist[i] / pTW * Math.log(normalizedHist[i] / pTW);
         }
       }
       hW[t] = hhW;
     } else {
       hW[t] = 0;
     }
   }

   // Find histogram index with maximum entropy
   // DAVID: ...where entropy[i] is defined to be (black_entropy[i] + white_entropy[i])
   double jMax = hB[0] + hW[0];
   int tMax = 0;
   for (int t = 1; t < hist.length; ++t) {
     double j = hB[t] + hW[t];
     if (j > jMax) {
       jMax = j;
       tMax = t;
     }
   }

   return tMax;
 }
}
