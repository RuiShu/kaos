from keras.layers import BatchNormalization
from keras.engine import Layer, InputSpec
from keras import initializations
from keras import backend as K

class BatchNormalization(BatchNormalization):
    @property
    def called_with(self):
        """Hack to always set called_with to None
        """
        return None

    @called_with.setter
    def called_with(self, value):
        pass
