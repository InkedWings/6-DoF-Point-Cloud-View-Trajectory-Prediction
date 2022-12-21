import numpy as np
import matplotlib.pyplot as plt

class ResultVisualization:
    """
    modelType:{GRU_Dense, GRU_En_De, Transformer}
    observ_step is the length of users' past history trajectory we used.
    pred_step is the length of prediction
    mode: Angular or Position
    """
    def __init__(self, mode, architecture, observ_step, pred_step):
        self.mode = mode
        self.architecture = architecture
        self.observ_step = observ_step
        self.pred_step = pred_step

    def loss_plot(self, train_loss, test_loss):
        plt.figure()
        plt.title("%s Train Test Loss for %s in period of %d points" % (self.architecture, self.mode, self.pred_step))
        epoch_num = len(train_loss)
        epochs = np.arange(1, epoch_num+1)
        plt.plot(epochs, train_loss, 'b', label='Train loss')
        plt.plot(epochs, test_loss, 'r', label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def data_plot(self, groundtruth, prediction):
        plt.figure()
        groundtruth_index = np.arange(0, self.observ_step + self.pred_step)
        # enc_output_index = np.arange(1, self.observ_step + 1)
        pred_index = np.arange(self.observ_step, self.observ_step + self.pred_step)
        if self.mode == "angle":
            subtitles = ["pitch", "yaw", "roll"]
        elif self.mode == "position":
            subtitles = ["x-axis", "y-axis", "z-axis"]

        Ylabel = 'degree' if self.mode == "Angular" else 'unit'
        plt.title("%s prediction for %s in period of %ds" % (self.architecture, self.mode, self.pred_step))
        plt.subplot(3, 1, 1)
        plt.title(subtitles[0])
        plt.plot(groundtruth_index, groundtruth[:, 0], 'b', label='GroundTruth')
        # plt.plot(enc_output_index, enc_output[:, 0], 'g', label = 'EncoderOutput')
        plt.plot(pred_index, prediction[:, 0], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.subplot(3, 1, 2)
        plt.title(subtitles[1])
        plt.plot(groundtruth_index, groundtruth[:, 1], 'b', label='GroundTruth')
        # plt.plot(enc_output_index, enc_output[:, 1], 'g', label='EncoderOutput')
        plt.plot(pred_index, prediction[:, 1], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.subplot(3, 1, 3)
        plt.title(subtitles[2])
        plt.plot(groundtruth_index, groundtruth[:, 2], 'b', label='GroundTruth')
        # plt.plot(enc_output_index, enc_output[:, 2], 'g', label='EncoderOutput')
        plt.plot(pred_index, prediction[:, 2], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.legend()
        plt.show()