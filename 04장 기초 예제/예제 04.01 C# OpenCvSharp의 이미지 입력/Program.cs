using System;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat src = Cv2.ImRead("logo.png", ImreadModes.ReducedColor2);
            Console.WriteLine(src);
        }
    }
}