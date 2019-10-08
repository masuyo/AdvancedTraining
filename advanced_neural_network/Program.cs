using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace advanced_neural_network
{
    public class Program
    {
        string[] trainData = File.ReadAllLines(@"...\Data\HusbandEvaluation.txt");
        AdvancedTraining app = new AdvancedTraining();

        void Run()
        {
            app.Train(trainData);
            FileTest();
            ConsoleTest();
        }

        void ConsoleTest()
        {
           
            while (true)
            {
                Console.Write("Age:");
                float age = float.Parse(Console.ReadLine());
                Console.Write("Height");
                float height = float.Parse(Console.ReadLine());
                Console.Write("Weight");
                float weight = float.Parse(Console.ReadLine());
                Console.Write("Salary");
                float salary = float.Parse(Console.ReadLine());

                Console.WriteLine("Prediction: " + app.Prediction(age, height, weight, salary));
            }
        }

        void FileTest()
        {
            int TP = 0, TN = 0, FP = 0, FN = 0;
            foreach (string line in trainData)
            {
                float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                int good = (int)values[4];
                int pred = (int)Math.Round(app.Prediction(values[0], values[1], values[2], values[3]));

                if (pred == good)
                {
                    if (pred == 1)
                    {
                        TP++;
                    }
                    else
                    {
                        TN++;
                    }
                }
                else
                {
                    if (pred == 1)
                    {
                        FP++;
                    }
                    else
                    {
                        FN++;
                    }
                }
                float accuracy = (float)(TP + TN) / (TP + FP + TN + FN);
                float precision = (float)TP / (TP + FP);
                float sensitivity = (float)TP / (TP + FN);
                float F1 = 2 * (precision * sensitivity) / (precision + sensitivity);
                Console.WriteLine(String.Format("True positive:\t{0}\nTrue negative:\t{1}\nFalse positive:\t{2}\nFalse negative:\t{3}", TP, TN, FP, FN));
                Console.WriteLine(String.Format("Accuracy:\t{0}\nPrecision:\t{1}\nSensitivity:\t{2}\nF1 score:\t{3}", accuracy, precision, sensitivity, F1));

            }
        }

        static void Main(string[] args)
        {
            new Program().Run();
        }
    }
}
