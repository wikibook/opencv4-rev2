using System;
using System.IO;
using System.Collections.Generic;
using OpenCvSharp;
using OpenCvSharp.ML;

namespace Project
{
    class Program
    {
        static Dictionary<int, string> labelDictionary = new Dictionary<int, string>()
        {
            { 0, "T-shirt/top" },
            { 1, "Trouser" },
            { 2, "Pullover" },
            { 3, "Dress" },
            { 4, "Coat" },
            { 5, "Sandal" },
            { 6, "Shirt" },
            { 7, "Sneaker" },
            { 8, "Bag" },
            { 9, "Ankle boot" }
        };

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

            KNearest knn = KNearest.Create();
            knn.Train(trainX, SampleTypes.RowSample, trainY);

            int count = 500;
            Mat results = new Mat();
            Mat neighborResponses = new Mat();
            Mat dists = new Mat();
            int retval = (int)knn.FindNearest(testX[0, count, 0, 784], 7, results, neighborResponses, dists);
            results.ConvertTo(results, MatType.CV_32S);

            for (int i = 0; i < count; ++i)
            {
                float[] imageArray = new float[784];
                Array.Copy(test.Item1, 784 * i, imageArray, 0, 784);
                Mat image = new Mat(28, 28, MatType.CV_32F, imageArray);
                image.ConvertTo(image, MatType.CV_8UC1);

                Console.WriteLine($"Index : {i}");
                Console.WriteLine($"예측값 : {labelDictionary[results.At<int>(i)]}");
                Console.WriteLine($"실젯값 : {labelDictionary[testY.At<int>(0, i)]}");
                Cv2.ImShow("image", image);
                Cv2.WaitKey(0);
            }
            Cv2.DestroyAllWindows();
        }
    }
}