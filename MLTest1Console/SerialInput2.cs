using Microsoft.ML.Data;

namespace MLTest1Console
{
    internal class SerialInput2
    {
        public float[] _labels = new float[32];
        [LoadColumn(0), ColumnName("dense_input")]
        public float[] labels
        { 
            get { return _labels;  }
            set { _labels = value; }
        }
    }
}