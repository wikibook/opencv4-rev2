using System;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat m = Mat.Eye(new Size(2, 2), MatType.CV_32FC3);

            Mat<Vec3f> mv3f = new Mat<Vec3f>(m);
            MatIndexer<Vec3f> indexer = mv3f.GetIndexer();
            //Mat.Indexer<Vec3f> indexer = m.GetGenericIndexer<Vec3f>();

            indexer[0, 1] = new Vec3f(2, 0, 0);

            for (int y = 0; y < m.Rows; y++)
            {
                for (int x = 0; x < m.Cols; x++)
                {
                    Console.WriteLine($"({y}, {x}) : {indexer[y, x].Item0} - {m.Get<Vec3f>(y, x).Item0}");
                }
            }

        }
    }
}