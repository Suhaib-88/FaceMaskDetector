from load_model.pytorch_loader import load_pytorch_model
import onnx
from torch.autograd import Variable
from onnx_tf.backend import prepare
import tensorflow as tf

#loading our trained Model
trained_model = load_pytorch_model('models/model360.pth')

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 3, 256, 256)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(trained_model, dummy_input, "output/mask.onnx")


## Loading ONNX Model ##
model= onnx.load('output/mask.onnx')
tf_rep= prepare(model)
print("inputs",tf_rep.inputs)
print("outputs",tf_rep.outputs)

## Exporting ONNX into TensorFlow Model ##
tf_rep.export_graph('output/mask.pb')


## Converting TensorFLow model into TensorFlow.lite Model for mobile app ##
converter=tf.compat.v1.lite.TFLiteConverter.from_saved_model('output/mask.pb')
tflite_model=converter.convert()
open('output/model.tflite','wb').write(tflite_model)