using System;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void DrawOpticalFlow(Mat frame, Mat flow)
        {
            Mat.Indexer<Vec2f> indexer = flow.GetGenericIndexer<Vec2f>();
            for (int y = 0; y < frame.Rows; y += 10)
            {
                for (int x = 0; x < frame.Cols; x += 10)
                {
                    Vec2f flowVec = indexer[y, x];
                    Point pt1 = new Point(x, y);
                    Point pt2 = new Point((int)(x + flowVec.Item0 * 5), (int)(y + flowVec.Item1 * 5));
                    Cv2.Line(frame, pt1, pt2, Scalar.Red, 2);
                    Cv2.Circle(frame, pt1, 1, Scalar.Blue, -1);
                }
            }
            Cv2.ImShow("Optical Flow (1)", frame);

            Mat[] vec = Cv2.Split(flow);
            Mat magnitude = new Mat();
            Mat angle = new Mat();
            Mat value = new Mat();
            Mat dst = new Mat();

            Cv2.CartToPolar(vec[0], vec[1], magnitude, angle, true);
            Cv2.Normalize(magnitude, magnitude, 0, 255, NormTypes.MinMax);
            magnitude.ConvertTo(magnitude, MatType.CV_8UC1);
            angle.ConvertTo(angle, MatType.CV_8UC1, 0.5);
            Cv2.Threshold(magnitude, value, 25, 255, ThresholdTypes.Binary);

            Cv2.Merge(new Mat[] { magnitude, angle, value }, dst);
            Cv2.CvtColor(dst, dst, ColorConversionCodes.HSV2BGR);
            Cv2.ImShow("Optical Flow (2)", dst);
        }

        static void Main(string[] args)
        {
            using (VideoCapture capture = new VideoCapture("car.mp4"))
            {
                Mat prevFrame = new Mat();
                Mat nextFrame = new Mat();
                Mat flow = new Mat();
                Mat dst = new Mat();

                while (true)
                {
                    capture.Read(nextFrame);
                    if (nextFrame.Empty())
                        break;

                    dst = nextFrame.Clone();
                    Cv2.CvtColor(nextFrame, nextFrame, ColorConversionCodes.BGR2GRAY);

                    if (prevFrame.Cols > 0)
                    {
                        Cv2.CalcOpticalFlowFarneback(prevFrame, nextFrame, flow, 0.5, 3, 15, 3, 5, 1.1, OpticalFlowFlags.FarnebackGaussian);
                        DrawOpticalFlow(dst, flow);
                    }
                    prevFrame = nextFrame.Clone();

                    if (Cv2.WaitKey(1) == 'q')
                        break;
                }
                Cv2.DestroyAllWindows();
            };
        }
    }
}