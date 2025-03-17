import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

import onnxruntime as rt

sess = rt.InferenceSession(
    "test.onnx")
    # , providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name_0 = sess.get_inputs()[0].name
input_name_1 = sess.get_inputs()[1].name
print(input_name_0)
print(input_name_1)
output_name = sess.get_outputs()[0].name
print(output_name)

X_test_0 = numpy.random.randn(1, 45).astype(numpy.float32)*0
X_test_1 = numpy.random.randn(1, 10, 45).astype(numpy.float32)*0

output_last = numpy.random.randn(1, 12).astype(numpy.float32)*0
X_test_1_tmp = numpy.random.randn(1, 9, 45).astype(numpy.float32)*0

X_test_0[0][5] = -1

# ortvalue_0 = rt.OrtValue.ortvalue_from_numpy(X_test_0)
# ortvalue_0.device_name()  # 'cpu'
# ortvalue_0.shape()        # shape of the numpy array X
# ortvalue_0.data_type()    # 'tensor(float)'
# ortvalue_0.is_tensor()    # 'True'
# numpy.array_equal(ortvalue_0.numpy(), X_test_0)  # 'True'

# ortvalue_1 = rt.OrtValue.ortvalue_from_numpy(X_test_1)
# ortvalue_1.device_name()  # 'cpu'
# ortvalue_1.shape()        # shape of the numpy array X
# ortvalue_1.data_type()    # 'tensor(float)'
# ortvalue_1.is_tensor()    # 'True'
# numpy.array_equal(ortvalue_1.numpy(), X_test_1)  # 'True'
options = rt.SessionOptions()
options.enable_profiling=True
for i in range(0,10) :
    X_test_0[0][33:45] = output_last

    pred_onx = sess.run(
        [output_name], {input_name_0 : X_test_0,
                  input_name_1 : X_test_1})[0]

    X_test_1_tmp[0][0:9] = X_test_1[0][1:10]
    X_test_1[0][0:9] = X_test_1_tmp[0][0:9]
    X_test_1[0][9] = X_test_0[0]
    output_last = pred_onx
    print(pred_onx)