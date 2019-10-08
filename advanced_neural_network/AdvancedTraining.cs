using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;

namespace advanced_neural_network
{
    class AdvancedTraining
    {
        const int inputSize = 4;
        const int hiddenNeuronCount = 3;
        const int outputSize = 1;

        readonly Variable x;
        readonly Function y;

        public AdvancedTraining()
        {
            x = Variable.InputVariable(new int[] { inputSize }, DataType.Float);
            Parameter w1 = new Parameter(new int[] { hiddenNeuronCount, inputSize }, DataType.Float,
                CNTKLib.GlorotNormalInitializer());
            Parameter b = new Parameter(new int[] { hiddenNeuronCount }, DataType.Float,
                CNTKLib.GlorotNormalInitializer());
            Parameter w2 = new Parameter(new int[] { outputSize, hiddenNeuronCount }, DataType.Float,
                CNTKLib.GlorotNormalInitializer());
            y = CNTKLib.Sigmoid(CNTKLib.Times(w2, CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w1, x), b))));
        }

        public void Train(string[] trainData)
        {
            int n = trainData.Length;
            Variable yt = Variable.InputVariable(new int[] { outputSize }, DataType.Float, "yt");
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt, "loss");

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            for (int i = 1; i <= 100; i++)
            {
                double sumLoss = 0;
                double sumEval = 0;
                foreach (string line in trainData)
                {
                    float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, LoadInput(values[0], values[1], values[2], values[3]) },
                        { yt, Value.CreateBatch(yt.Shape, new float[] { values[4] },
                            DeviceDescriptor.CPUDevice)}
                    };
                    var outputDataMap = new Dictionary<Variable, Value>() { { loss, null } };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage();
                }
                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", i, sumLoss / n, sumEval / n));
            }
        }

        Value LoadInput(float age, float height, float weight, float salary)
        {
            float[] x_store = new float[inputSize];
            x_store[0] = age / 100;
            x_store[1] = height / 250;
            x_store[2] = weight / 150;
            x_store[3] = salary / 150000;
            return Value.CreateBatch(x.Shape, x_store, DeviceDescriptor.CPUDevice);
        }

        public float Prediction(float age, float height, float weight, float salary)
        {
            var inputDataMap = new Dictionary<Variable, Value>() { { x, LoadInput(age, height, weight, salary) } };
            var outputDataMap = new Dictionary<Variable, Value>() { { y, null } };
            y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
            return outputDataMap[y].GetDenseData<float>(y)[0][0];
        }

    }
}
