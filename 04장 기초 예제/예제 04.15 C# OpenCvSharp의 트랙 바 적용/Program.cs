using System;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Cv2.NamedWindow("Palette");
            Cv2.CreateTrackbar("Color", "Palette", 255);

            while (true)
            {
                int pixel = Cv2.GetTrackbarPos("Color", "Palette");
                Mat src = new Mat(new Size(500, 500), MatType.CV_8UC3, new Scalar(pixel, pixel, pixel));

                Cv2.ImShow("Palette", src);
                if (Cv2.WaitKey(33) == 'q')
                    break;
            }

            Cv2.DestroyAllWindows();
        }
    }
}
