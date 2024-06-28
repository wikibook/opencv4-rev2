using System;
using System.IO;
using System.Linq;
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

        static float[] HogCompute(float[] images)
        {
            HOGDescriptor hog = new HOGDescriptor(new Size(28, 28), new Size(8, 8), new Size(4, 4), new Size(4, 4), 9, 1, -1, HistogramNormType.L2Hys, 0.2, true, 28);
            List<float[]> descriptor = new List<float[]>();

            for (int num = 0; num < images.Length / 784; num++)
            {
                float[] imageArray = new float[784];
                Array.Copy(images, 784 * num, imageArray, 0, 784);
                Mat image = new Mat(28, 28, MatType.CV_32F, imageArray);
                image.ConvertTo(image, MatType.CV_8UC1);
                descriptor.Add(hog.Compute(image));
            }

            List<float> flattenDescriptor = (from list in descriptor from item in list select item).ToList();
            return flattenDescriptor.ToArray();
        }

        static void Main(string[] args)
        {
            Tuple<float[], int[]> train = LoadTrainData("./fashion-mnist/train-images-idx3-ubyte", "./fashion-mnist/train-labels-idx1-ubyte", 60000);
            Tuple<float[], int[]> test = LoadTrainData("./fashion-mnist/t10k-images-idx3-ubyte", "./fashion-mnist/t10k-labels-idx1-ubyte", 10000);

            float[] trainDescriptor = HogCompute(train.Item1);
            float[] testDescriptor = HogCompute(test.Item1);

            Mat trainX = new Mat(60000, trainDescriptor.Length / 60000, MatType.CV_32F, trainDescriptor);
            Mat trainY = new Mat(1, 60000, MatType.CV_32S, train.Item2);
            Mat testX = new Mat(10000, testDescriptor.Length / 10000, MatType.CV_32F, testDescriptor);
            Mat testY = new Mat(1, 10000, MatType.CV_32S, test.Item2);

            SVM svm = SVM.Create();
            svm.Type = SVM.Types.CSvc;
            svm.KernelType = SVM.KernelTypes.Rbf;
            svm.Gamma = 0.5;
            svm.C = 0.5;
            svm.Train(trainX, SampleTypes.RowSample, trainY);

            int count = 500;
            Mat results = new Mat();
            svm.Predict(testX[0, count, 0, testX.Width], results);
            results.ConvertTo(results, MatType.CV_32S);

            Mat matches = new Mat();
            Cv2.Compare(results, testY[0, 1, 0, count].T(), matches, CmpType.EQ);
            Console.WriteLine((float)Cv2.CountNonZero(matches) / count * 100);
        }
    }
}