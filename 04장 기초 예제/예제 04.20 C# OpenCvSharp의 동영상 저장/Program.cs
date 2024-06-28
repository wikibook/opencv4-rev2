using System;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            VideoCapture capture = new VideoCapture("star.mp4");
            Mat frame = new Mat(new Size(capture.FrameWidth, capture.FrameHeight), MatType.CV_8UC3);
            VideoWriter videoWriter = new VideoWriter();
            bool isWrite = false;

            while (true)
            {
                if (capture.PosFrames == capture.FrameCount)
                    capture.Open("star.mp4");

                capture.Read(frame);
                Cv2.ImShow("VideoFrame", frame);

                int key = Cv2.WaitKey(33);
                if (key == 4)
                {
                    videoWriter.Open("video.avi", FourCC.XVID, 30, frame.Size(), true);
                    isWrite = true;

                }
                else if (key == 24)
                {
                    videoWriter.Release();
                    isWrite = false;
                }
                else if (key == 'q')
                    break;

                if (isWrite == true)
                    videoWriter.Write(frame);
            }

            videoWriter.Release();
            capture.Release();
            Cv2.DestroyAllWindows();
        }
    }
}