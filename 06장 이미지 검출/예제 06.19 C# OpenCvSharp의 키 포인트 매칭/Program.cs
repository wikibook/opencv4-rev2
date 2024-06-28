using System;
using System.Linq;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat query = Cv2.ImRead("query.jpg");
            Mat train = Cv2.ImRead("train.jpg");
            Mat queryGray = new Mat();
            Mat trainGray = new Mat();
            Cv2.CvtColor(query, queryGray, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(train, trainGray, ColorConversionCodes.BGR2GRAY);

            ORB orb = ORB.Create(5000);

            KeyPoint[] kp1, kp2;
            Mat des1 = new Mat(), des2 = new Mat();
            orb.DetectAndCompute(queryGray, null, out kp1, des1);
            orb.DetectAndCompute(trainGray, null, out kp2, des2);

            BFMatcher bf = new BFMatcher(NormTypes.Hamming, true);
            DMatch[] matches = bf.Match(des1, des2);
            Array.Sort(matches, (x, y) => x.Distance.CompareTo(y.Distance));

            int count = 100;
            for (int i = 0; i < count; i++)
            {
                int idx = matches[i].QueryIdx;
                Point2f pt = kp1[idx].Pt;
                Cv2.Circle(query, (int)pt.X, (int)pt.Y, 3, new Scalar(0, 0, 255), 3);
            }

            DrawMatchesFlags flag = DrawMatchesFlags.NotDrawSinglePoints | DrawMatchesFlags.DrawRichKeypoints;
            Mat result = new Mat();
            Cv2.DrawMatches(query, kp1, train, kp2, matches.Take(count).ToArray(), result, flags: flag);

            Cv2.ImShow("Matching", result);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}