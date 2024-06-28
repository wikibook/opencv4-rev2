using System;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat image = Cv2.ImRead("qr-code.png");

            QRCodeDetector detector = new QRCodeDetector();
            bool retvalDetect = detector.DetectMulti(image, out Point2f[] points);
            bool retvalDecode = detector.DecodeMulti(image, points, out string[] decodedInfo);

            for (int i=0; i<decodedInfo.Length; i++)
            {
                Console.WriteLine(decodedInfo[i]);
                Cv2.Rectangle(
                    image,
                    new Point(points[4 * i].X, points[4 * i].Y),
                    new Point(points[4 * i + 2].X, points[4 * i + 2].Y),
                    new Scalar(0, 255, 0),
                    4
                );
            }

            Cv2.ImShow("image", image);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}