using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLTest1Console
{
    class Program
    {
        static void Main(string[] args)
        {
            string serial = " ";
            MLContext context = new MLContext();

            var modelName = "model8s.onnx";
            var emptyData = new List<SerialInput>();
            var data = context.Data.LoadFromEnumerable(emptyData);

            // Keras define la capa de entrada como un array de doubles
            // Asi que concateno todas las columnas de entrada que son features
            // en un solo array
            var pipeline = context.Transforms.Concatenate("dense_9_input",
                new[] { "d1","d2","d3","d4","d5","d6","d7","d8","d9","d10",
                "d11","d12","d13","d14","d15","d16","d17","d18","d19","d20",
                "d21","d22","d23","d24","d25","d26","d27","d28","d29","d30",
                "d31","d32" })
                .Append(context.Transforms.ApplyOnnxModel(modelName));

            // Como el modelo ya está entrenado, no es necesario entrenarlo de nuevo
            // pero si hay que ejecutar Fit, así que lo ejecuto con un conjunto de 
            // datos vacios
            var model = pipeline.Fit(data);


            // Ahora si podemos hacer algunas predicciones, los valores de pertenencia para
            // decidir si un resultado es válido o no los decidí al azar, 50 y 50 de probabilidad
            var predictionEngine = context.Model.CreatePredictionEngine<SerialInput, SerialPrediction>(model);

            
            while (serial != "")
            {
                serial = "";
                Console.Write("Ingrese un número de serie de 32 digitos: ");
                serial = Console.ReadLine();

                if (serial == "")
                {
                    break;
                }
                if (serial.All(char.IsDigit) && serial.Length == 32)
                {
                    var s = StrToSerialInput(serial);
                    var prediction = predictionEngine.Predict(s);
                    if (prediction.Score[0] >= 0.5)
                    {
                        Console.WriteLine($"El serial parece ser válido {prediction.Score[0],10:F5}");
                    }
                    else
                    {
                        Console.WriteLine($"El serial parece ser inválido {prediction.Score[0],10:F5}");
                    }
                    
                }
                else 
                {
                    Console.WriteLine("En serial debe ser de 32 dígitos");
                }
                var t = Console.ReadLine();
            }



            
        }
        /// <summary>
        /// Convierte un numero de serie que ingresa por consola como una cadena
        /// a un array de double
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static SerialInput StrToSerialInput(string s)
        {
            SerialInput ret = new SerialInput();

            for (int i = 0; i < 32; i++)
            {
                ret.GetType().GetProperty($"d{(i+1).ToString()}").SetValue(ret, (float)char.GetNumericValue(s[i]));
            }
            return ret;
        }
    }
}
