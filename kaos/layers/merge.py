from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Concat(Layer):
    def __init__(self, concat_axis):
        self.concat_axis = concat_axis
        super(Concat, self).__init__()

    def call(self, inputs, mask=None):
        return K.concatenate(inputs, self.concat_axis)

    def _validate_input_shape(self, input_shape):
        axes = zip(*input_shape)
        del axes[self.concat_axis]
        mismatch = any([dims.count(dims[0]) != len(dims) for dims in axes])
        if mismatch:
            raise Exception("Only works if input shapes match in all non-concat axes")

    def get_output_shape_for(self, input_shape):
        self._validate_input_shape(input_shape)
        axes = zip(*input_shape)
        output_shape = [dims[0] for dims in axes]
        output_shape[self.concat_axis] = sum(axes[self.concat_axis])
        return tuple(output_shape)
