"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape


class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """

    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(
                parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        PATCH_SIZE = 64
        vol = med_reshape(volume, new_shape=(
            volume.shape[0], PATCH_SIZE, PATCH_SIZE))

        pred_vol = self.single_volume_inference(vol)

        return pred_vol

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the 
        # X (0th) dimension. After that, put all slices into a 3D Numpy array. 
        # You can verify if your method is correct by running it on one of the
        # volumes in your training set and comparing with the label in 3D Slicer.
        volume = volume.astype(np.single)/np.amax(volume)
        pred_vol = np.zeros(volume.shape)

        for slc_ind in range(volume.shape[0]):
            img = torch.from_numpy(volume[slc_ind, :, :]).unsqueeze(
                0).unsqueeze(0).to(self.device, dtype=torch.float)
            pred = self.model(img)
            mask = torch.argmax(np.squeeze(pred.cpu().detach()), dim=0).numpy()
            pred_vol[slc_ind, :, :] = mask

        return pred_vol
