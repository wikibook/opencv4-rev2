using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat src = Cv2.ImRead("ferris-wheel.jpg");
            List<Mat> gaussianPyramid = new List<Mat>();
            List<Mat> laplacianPyramid = new List<Mat>();
            List<Size> sizes = new List<Size>();

            int numLevels = 4;
            Mat temp = src.Clone();
            for (int i = 0; i < numLevels; i++)
            {
                Mat down = new Mat();
                Cv2.PyrDown(temp, down);
                gaussianPyramid.Add(down);
                temp = down.Clone();
                sizes.Add(down.Size());
            }

            for (int i = 0; i < numLevels - 1; i++)
            {
                Mat up = new Mat();
                Mat laplacian = new Mat();
                Cv2.PyrUp(gaussianPyramid[i + 1], up, sizes[i]);
                Cv2.Subtract(gaussianPyramid[i], up, laplacian);
                laplacianPyramid.Add(laplacian);
            }

            Cv2.ImShow("gaussianPyramid_0", gaussianPyramid[0]);
            Cv2.ImShow("laplacianPyramid_0", laplacianPyramid[0]);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}
