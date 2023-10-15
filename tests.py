import unittest
from streamlit_app import load_LSTM_model, load_CNN_model

class TestModelLoading(unittest.TestCase):

    def test_load_LSTM_model(self):
        lstm_model = load_LSTM_model()
        self.assertIsNotNone(lstm_model)
        # Add more assertions if needed

    def test_load_CNN_model(self):
        cnn_model = load_CNN_model()
        self.assertIsNotNone(cnn_model)
        # Add more assertions if needed

if __name__ == '__main__':
    unittest.main()
