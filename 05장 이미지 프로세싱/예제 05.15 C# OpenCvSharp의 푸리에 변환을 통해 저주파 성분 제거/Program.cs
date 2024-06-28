using System;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Shift(Mat img)
        {
            int cx = img.Cols / 2;
            int cy = img.Rows / 2;
            Mat q0 = new Mat(img, new Rect(0, 0, cx, cy));
            Mat q1 = new Mat(img, new Rect(cx, 0, cx, cy));
            Mat q2 = new Mat(img, new Rect(0, cy, cx, cy));
            Mat q3 = new Mat(img, new Rect(cx, cy, cx, cy));

            Mat tmp = new Mat();
            q0.CopyTo(tmp);
            q3.CopyTo(q0);
            tmp.CopyTo(q3);

            q1.CopyTo(tmp);
            q2.CopyTo(q1);
            tmp.CopyTo(q2);
        }

        static Mat FourierSpectrum(Mat dft)
        {
            Mat[] dftPlanes;
            Cv2.Split(dft, out dftPlanes);

            Mat spectrum = new Mat();
            Cv2.Magnitude(dftPlanes[0], dftPlanes[1], spectrum);

            Shift(spectrum);
            spectrum += Scalar.All(1);
            Cv2.Log(spectrum, spectrum);

            Cv2.Normalize(spectrum, spectrum, 0, 255, NormTypes.MinMax);
            spectrum.ConvertTo(spectrum, MatType.CV_8UC1);
            return spectrum;
        }

        static void Main(string[] args)
        {
            Mat src = Cv2.ImRead("pears.jpg", ImreadModes.Grayscale);
            Mat src32F = new Mat();
            Mat dft = new Mat();

            src.ConvertTo(src32F, MatType.CV_32F);
            Cv2.Dft(src32F, dft, DftFlags.ComplexOutput);
            Mat spectrum = FourierSpectrum(dft);

            Shift(dft);
            int d = 10;
            int cx = dft.Cols / 2;
            int cy = dft.Rows / 2;
            Cv2.Rectangle(dft, new Rect(cx - d, cy - d, 2 * d, 2 * d), Scalar.All(0), -1);
            Shift(dft);

            Mat idft = new Mat();
            Mat dst = new Mat();

            Cv2.Idft(dft, idft);
            Mat[] magnitude = Cv2.Split(idft);
            Cv2.Magnitude(magnitude[0], magnitude[1], dst);
            Cv2.Normalize(dst, dst, 0, 255, NormTypes.MinMax);
            dst.ConvertTo(dst, MatType.CV_8UC1);

            Cv2.ImShow("spectrum", spectrum);
            Cv2.ImShow("dst", dst);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}
