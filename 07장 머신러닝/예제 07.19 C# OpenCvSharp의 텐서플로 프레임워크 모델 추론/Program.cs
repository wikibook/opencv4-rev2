using System;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            const string config = "tensorflow_model/graph.pbtxt";
            const string model = "tensorflow_model/frozen_inference_graph.pb";
            string[] classNames = File.ReadAllLines("tensorflow_model/labelmap.txt");

            Mat image = new Mat("bus.jpg");
            Net net = Net.ReadNetFromTensorflow(model, config);
            Mat inputBlob = CvDnn.BlobFromImage(image, swapRB: true, crop: false);
            
            net.SetInput(inputBlob);
            string[] outBlobNames = new string[] { "detection_out_final", "detection_masks" };
            Mat[] outputBlobs = outBlobNames.Select(toMat => new Mat()).ToArray();

            net.Forward(outputBlobs, outBlobNames);
            Mat boxes = new Mat(outputBlobs[0].Size(2), outputBlobs[0].Size(3), MatType.CV_32F, outputBlobs[0].Ptr(0));
            Mat masks = outputBlobs[1];

            int height = image.Rows;
            int width = image.Cols;
            double threshold = 0.9;
            for (int idx = 0; idx < boxes.Rows; idx++)
            {
                int classID = (int)boxes.At<float>(idx, 1);
                double confidence = boxes.At<float>(idx, 2);
                string label = classNames[classID];

                if (confidence > threshold)
                {
                    int x1 = (int)(boxes.At<float>(idx, 3) * image.Width);
                    int y1 = (int)(boxes.At<float>(idx, 4) * image.Height);
                    int x2 = (int)(boxes.At<float>(idx, 5) * image.Width);
                    int y2 = (int)(boxes.At<float>(idx, 6) * image.Height);

                    Mat mask = masks.Row(idx).Col(classID).Reshape(1, masks.Size(2));
                    Cv2.Resize(mask, mask, new Size(x2 - x1, y2 - y1), interpolation: InterpolationFlags.Nearest);
                    Cv2.Compare(mask, threshold, mask, CmpType.GT);

                    Mat color = new Mat(mask.Size(), MatType.CV_8UC3, new Scalar(255, 0, 0));
                    Mat colorMask = new Mat(color.Size(), MatType.CV_8UC3);
                    Cv2.BitwiseAnd(color, color, colorMask, mask);

                    Mat roi = new Mat(image, new Rect(x1, y1, x2 - x1, y2 - y1));
                    Cv2.AddWeighted(roi, 1.0, colorMask, 1.0, 0.0, roi);
                    
                    Cv2.Rectangle(image, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255));
                    Cv2.PutText(image, label, new Point(x1, y1), HersheyFonts.HersheyComplex, 1.0, Scalar.Red);
                }
            }
            Cv2.ImShow("image", image);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}