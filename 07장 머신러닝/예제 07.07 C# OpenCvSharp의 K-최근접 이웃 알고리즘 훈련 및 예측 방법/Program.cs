using System;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.ML;

namespace Project
{
    class Program
    {
        static Tuple<float[], int[]> LoadTrainData(string imagePath, string labelPath, int length)
        {
            using (FileStream imageData = new FileStream(imagePath, FileMode.Open))
            using (FileStream labelData = new FileStream(labelPath, FileMode.Open))
            using (BinaryReader imageBinary = new BinaryReader(imageData))
            using (BinaryReader labelBinary = new BinaryReader(labelData))
            {
                imageBinary.ReadBytes(16);
                labelBinary.ReadBytes(8);

                float[] image = new float[length * 784];
                int[] label = new int[length];

                for (int dataIndex = 0; dataIndex < length; ++dataIndex)
                {
                    for (int i = 0; i < 784; ++i)
                    {
                        byte img = imageBinary.ReadByte();
                        image[dataIndex * 784 + i] = (float)img;
                    }
                    byte lb = labelBinary.ReadByte();
                    label[dataIndex] = (int)lb;
                }
                return new Tuple<float[], int[]>(image, label);
            }
        }

        static void Main(string[] args)
        {
            Tuple<float[], int[]> train = LoadTrainData("./fashion-mnist/train-images-idx3-ubyte", "./fashion-mnist/train-labels-idx1-ubyte", 60000);
            Tuple<float[], int[]> test = LoadTrainData("./fashion-mnist/t10k-images-idx3-ubyte", "./fashion-mnist/t10k-labels-idx1-ubyte", 10000);

            Mat trainX = new Mat(60000, 784, MatType.CV_32F, train.Item1);
            Mat trainY = new Mat(1, 60000, MatType.CV_32S, train.Item2);
            Mat testX = new Mat(10000, 784, MatType.CV_32F, test.Item1);
            Mat testY = new Mat(1, 10000, MatType.CV_32S, test.Item2);

            int num = 0;
            float[] imageArray = new float[784];
            Array.Copy(train.Item1, 784 * num, imageArray, 0, 784);
            Mat image = new Mat(28, 28, MatType.CV_32F, imageArray);
            image.ConvertTo(image, MatType.CV_8UC1);

            KNearest knn = KNearest.Create();
            knn.Train(trainX, SampleTypes.RowSample, trainY);

            int count = 500;
            Mat results = new Mat();
            Mat neighborResponses = new Mat();
            Mat dists = new Mat();
            int retval = (int)knn.FindNearest(testX[0, count, 0, 784], 7, results, neighborResponses, dists);
            results.ConvertTo(results, MatType.CV_32S);

            Mat matches = new Mat();
            Cv2.Compare(results, testY[0, 1, 0, count].T(), matches, CmpType.EQ);
            Console.WriteLine((float)Cv2.CountNonZero(matches) / count * 100);
        }
    }
}