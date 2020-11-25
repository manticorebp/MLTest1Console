using Microsoft.ML.Data;

namespace MLTest1Console
{
    internal class SerialPrediction
    {
        [ColumnName("dense_10")]
        public float[] Score { get; set; }
    }
}