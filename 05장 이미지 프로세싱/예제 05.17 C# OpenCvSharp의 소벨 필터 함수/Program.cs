using System;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat src = Cv2.ImRead("book.jpg", ImreadModes.Grayscale);
            Mat kernel = new Mat(3, 3, MatType.CV_32FC1, new float[] {
                -1, 0, 1,
                -2, 0, 2,
                -1, 0, 1
            });
            Mat dst1 = new Mat();
            Mat dst2 = new Mat();
            Mat dst3 = new Mat();

            Cv2.Filter2D(src, dst1, MatType.CV_8UC1, kernel);
            Cv2.Sobel(src, dst2, MatType.CV_8UC1, 1, 0, 3);
            Cv2.Compare(dst1, dst2, dst3, CmpType.EQ);

            Cv2.ImShow("dst1", dst1);
            // Cv2.ImShow("dst2", dst2);
            // Cv2.ImShow("dst3", dst3);

            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}
