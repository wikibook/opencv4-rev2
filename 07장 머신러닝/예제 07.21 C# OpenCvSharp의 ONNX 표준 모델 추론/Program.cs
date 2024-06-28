using System;
using System.Linq;
using System.Collections.Generic;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat src = Cv2.ImRead("crowd-of-people.jpg");

            int height = src.Height;
            int width = src.Width;
            int inputW = 640;
            int inputH = 640;
            int[] strides = { 8, 16, 32 };

            Net net = CvDnn.ReadNetFromOnnx("onnx_model/yunet.onnx");
            Mat inputBlob = CvDnn.BlobFromImage(src, 1.0, new Size(inputW, inputH), swapRB:false, crop:false);

            net.SetInput(inputBlob);
            string[] outBlobNames = net.GetUnconnectedOutLayersNames();
            Array.Sort(outBlobNames, (x, y) =>
            {
                string[] xParts = x.Split('_');
                string[] yParts = y.Split('_');
                return xParts[0] == yParts[0]
                    ? int.Parse(xParts[1]) - int.Parse(yParts[1])
                    : string.Compare(xParts[0], yParts[0], StringComparison.Ordinal);
            });
            Mat[] outputBlobs = outBlobNames.Select(toMat => new Mat()).ToArray();
            net.Forward(outputBlobs, outBlobNames);

            Mat[] bbox = new Mat[3] { outputBlobs[0], outputBlobs[1], outputBlobs[2] };
            Mat[] classes = new Mat[3] { outputBlobs[3], outputBlobs[4], outputBlobs[5] };
            Mat[] kps = new Mat[3] { outputBlobs[6], outputBlobs[7], outputBlobs[8] };
            Mat[] objectness = new Mat[3] { outputBlobs[9], outputBlobs[10], outputBlobs[11] };

            List<Rect> faces = new List<Rect>();
            List<Point[]> landmarks = new List<Point[]>();
            List<float> scores = new List<float>();

            for (int i = 0; i < strides.Length; i++)
            {
                int rows = inputH / strides[i];
                int cols = inputW / strides[i];

                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        int idx = r * cols + c;

                        float clsScore = classes[i].At<float>(0, idx, 0);
                        float objScore = objectness[i].At<float>(0, idx, 0);
                        float score = (float)Math.Sqrt(clsScore * objScore);

                        float[] box = new float[4]
                        {
                            bbox[i].At<float>(0, idx, 0),
                            bbox[i].At<float>(0, idx, 1),
                            bbox[i].At<float>(0, idx, 2),
                            bbox[i].At<float>(0, idx, 3),
                        };

                        float[] kp = new float[10]
                        {
                            kps[i].At<float>(0, idx, 0),
                            kps[i].At<float>(0, idx, 1),
                            kps[i].At<float>(0, idx, 2),
                            kps[i].At<float>(0, idx, 3),
                            kps[i].At<float>(0, idx, 4),
                            kps[i].At<float>(0, idx, 5),
                            kps[i].At<float>(0, idx, 6),
                            kps[i].At<float>(0, idx, 7),
                            kps[i].At<float>(0, idx, 8),
                            kps[i].At<float>(0, idx, 9),
                        };

                        float cx = (c + box[0]) * strides[i] / inputW * width;
                        float cy = (r + box[1]) * strides[i] / inputH * height;
                        float w = (float)(Math.Exp(box[2]) * strides[i] / inputW * width);
                        float h = (float)(Math.Exp(box[3]) * strides[i] / inputH * height);

                        float x1 = cx - w / 2.0f;
                        float y1 = cy - h / 2.0f;

                        float rex = (c + kp[0]) * strides[i] / inputW * width;
                        float rey = (r + kp[1]) * strides[i] / inputH * height;
                        float lex = (c + kp[2]) * strides[i] / inputW * width;
                        float ley = (r + kp[3]) * strides[i] / inputH * height;

                        float ntx = (c + kp[4]) * strides[i] / inputW * width;
                        float nty = (r + kp[5]) * strides[i] / inputH * height;

                        float rcmx = (c + kp[6]) * strides[i] / inputW * width;
                        float rcmy = (r + kp[7]) * strides[i] / inputH * height;
                        float lcmx = (c + kp[8]) * strides[i] / inputW * width;
                        float lcmy = (r + kp[9]) * strides[i] / inputH * height;

                        scores.Add(score);
                        faces.Add(new Rect((int)x1, (int)y1, (int)w, (int)h));
                        landmarks.Add(new Point[] {
                            new Point((int)rex, (int)rey),
                            new Point((int)lex, (int)ley),
                            new Point((int)ntx, (int)nty),
                            new Point((int)rcmx, (int)rcmy),
                            new Point((int)lcmx, (int)lcmy) 
                        });
                    }
                }
            }

            float scoreThreshold = 0.7f;
            float nmsThreshold = 0.4f;
            CvDnn.NMSBoxes(faces, scores, scoreThreshold, nmsThreshold, out int[] indices);

            foreach (int i in indices)
            {
                Rect rect = faces[i];
                Point[] landmark = landmarks[i];

                Cv2.Rectangle(src, rect, Scalar.Red);
                Cv2.Circle(src, landmark[0], 3, Scalar.Yellow, 2);
                Cv2.Circle(src, landmark[1], 3, Scalar.Yellow, 2);
                Cv2.Circle(src, landmark[2], 3, Scalar.Magenta, 2);
            }
            Cv2.ImShow("src", src);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
    }
}