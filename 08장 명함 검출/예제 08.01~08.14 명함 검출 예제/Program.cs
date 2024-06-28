using System;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Text;

namespace Project
{
    class Program
    {
        static double CalcAngle(Point pt1, Point pt0, Point pt2)
        {
            double u1 = pt1.X - pt0.X, u2 = pt1.Y - pt0.Y;
            double v1 = pt2.X - pt0.X, v2 = pt2.Y - pt0.Y;

            double numerator = u1 * v1 + u2 * v2;
            double denominator = Math.Sqrt(u1 * u1 + u2 * u2) * Math.Sqrt(v1 * v1 + v2 * v2);

            return numerator / denominator;
        }

        public static Point[] FindSquare(Mat src)
        {
            Mat[] split = Cv2.Split(src);
            Mat blur = new Mat();
            Mat binary = new Mat();
            Point[] square = new Point[4];

            int N = 10;
            double cos = 1;
            double max = src.Size().Width * src.Size().Height * 0.9;
            double min = src.Size().Width * src.Size().Height * 0.1;

            for (int channel = 0; channel < 3; channel++)
            {
                Cv2.GaussianBlur(split[channel], blur, new Size(5, 5), 1);
                for (int i = 0; i < N; i++)
                {
                    Cv2.Threshold(blur, binary, i * 255 / N, 255, ThresholdTypes.Binary);

                    Point[][] contours;
                    HierarchyIndex[] hierarchy;
                    Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxTC89KCOS);

                    for (int j = 0; j < contours.Length; j++)
                    {
                        double perimeter = Cv2.ArcLength(contours[j], true);
                        Point[] result = Cv2.ApproxPolyDP(contours[j], perimeter * 0.02, true);

                        double area = Cv2.ContourArea(result);
                        bool convex = Cv2.IsContourConvex(result);

                        if (result.Length == 4 && area > min && area < max && convex)
                        {
                            double[] angles = new double[4];
                            for (int k = 1; k < 5; k++)
                            {
                                double angle = Math.Abs(CalcAngle(result[(k - 1) % 4], result[k % 4], result[(k + 1) % 4]));
                                angles[k - 1] = angle;
                            }
                            if (angles.Max() < cos && angles.Max() < 0.15)
                            {
                                cos = angles.Max();
                                square = result;
                            }
                        }
                    }
                }
            }
            return square;
        }

        public static Mat DrawSquare(Mat src, Point[] square)
        {
            Mat drawSquare = src.Clone();
            Point[][] pts = new Point[][] { square };
            Cv2.Polylines(drawSquare, pts, true, Scalar.Yellow, 3, LineTypes.AntiAlias, 0);
            return drawSquare;
        }

        public static Mat PerspectiveTransform(Mat src, Point[] square)
        {
            Mat dst = new Mat();
            Moments moments = Cv2.Moments(square);
            double cX = moments.M10 / moments.M00;
            double cY = moments.M01 / moments.M00;

            Point2f[] srcPts = new Point2f[4];
            for (int i = 0; i < square.Length; i++)
            {
                if (cX > square[i].X && cY > square[i].Y) srcPts[0] = square[i];
                if (cX > square[i].X && cY < square[i].Y) srcPts[1] = square[i];
                if (cX < square[i].X && cY > square[i].Y) srcPts[2] = square[i];
                if (cX < square[i].X && cY < square[i].Y) srcPts[3] = square[i];
            }
            Point2f[] dstPts = new Point2f[4]
            {
                new Point2f(0, 0),
                new Point2f(0, src.Height),
                new Point2f(src.Width, 0),
                new Point2f(src.Width, src.Height)
            };

            Mat matrix = Cv2.GetPerspectiveTransform(srcPts, dstPts);
            Cv2.WarpPerspective(src, dst, matrix, new Size(src.Width, src.Height));
            return dst;
        }

        public static string OCR(Mat src, string datapath, string language)
        {
            OCRTesseract ocr = OCRTesseract.Create(datapath, language);
            ocr.Run(src, out string outputText, out Rect[] componentRects, out string[] componentTexts, out float[] componentConfidences, ComponentLevels.Word);

            Console.WriteLine("outputText:");
            Console.WriteLine(outputText);

            Console.WriteLine("componentRects:");
            foreach (var componentRect in componentRects)
            {
                Console.WriteLine(componentRect);
            }

            Console.WriteLine("componentTexts:");
            foreach (var componentRect in componentTexts)
            {
                Console.WriteLine(componentRect);
            }

            Console.WriteLine("componentConfidences:");
            foreach (var componentRect in componentConfidences)
            {
                Console.WriteLine(componentRect);
            }

            return outputText;
        }

        static void Main(string[] args)
        {
            Mat src = Cv2.ImRead("card.png", ImreadModes.ReducedColor2);

            Point[] square = FindSquare(src);
            Mat dst = PerspectiveTransform(src, square);
            string outputText = OCR(dst, "../../tessdata", "eng");

            Cv2.ImShow("dst", dst);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}
