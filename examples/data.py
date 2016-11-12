from kaos.data import MnistLoader

class MnistVAE(MnistLoader):
    def __init__(self, batchsize):
        super(MnistVAE, self).__init__(batchsize)
        self.binarize()

    def get_training_batch(self):
        x, y = super(MnistVAE, self).get_training_batch()
        return x, x

    def get_training_set(self):
        x, y = super(MnistVAE, self).get_training_set()
        return x, x

    def get_validation_set(self):
        x, y = super(MnistVAE, self).get_validation_set()
        return x, x

    def get_test_set(self):
        x, y = super(MnistVAE, self).get_test_set()
        return x, x
