import SimpleITK
import numpy as np
import torch
from scipy import ndimage
from monai.inferers import sliding_window_inference
from uncertainty import ensemble_uncertainties_classification
from pathlib import Path
from unet import *
from density_unet import *
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from monai.data import decollate_batch


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        # print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a 
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2


class Baseline(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        output_path = Path("/output/images/")
        if not output_path.exists():
            output_path.mkdir()

        # self._input_path = Path("/input/images/brain-mri/")
        self._segmentation_output_path = Path("/output/images/white-matter-multiple-sclerosis-lesion-segmentation/")
        self._uncertainty_output_path = Path("/output/images/white-matter-multiple-sclerosis-lesion-uncertainty-map/")

        # self._segmentation_output_path = Path("/output/segmentation/")
        # self._uncertainty_output_path = Path("/output/uncertainty/")

        self.device = get_default_device()

        self.Ke = 1
        models = []
        roi_size = (96, 96, 96)
        in_channels, out_channels = 1, 2
        in_shape = (in_channels,) + roi_size
        unet = PytorchUNet3D(in_shape,
                             c=None,
                             norm_layer=ActNormLP3D,
                             num_classes=out_channels,
                             n_channels=in_channels,
                             device=self.device,
                             cout=None).to(self.device)
        x = torch.rand((1, 1,) + roi_size).to(self.device)
        y = unet(x)

        super_model = DensityUnet(path_gmms=['_', '_', '_', '_', './gmm1.pth'], path_density_unet='./model1.pth',
                                  unet=unet,
                                  device=self.device,
                                  combination='last', K=4).to(self.device)
        super_model.Unet_init = True
        super_model.eval()

        self.super_model = super_model
        self.act = torch.nn.Softmax(dim=1)
        self.th = 0.35
        self.roi_size = (96, 96, 96)
        self.sw_batch_size = 4

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_map, uncertainty_map = self.predict(input_image=input_image)

        # Write resulting segmentation to output location
        segmentation_path = self._segmentation_output_path / input_image_file_path.name
        if not self._segmentation_output_path.exists():
            self._segmentation_output_path.mkdir()
        SimpleITK.WriteImage(segmented_map, str(segmentation_path), True)

        # Write resulting uncertainty map to output location
        uncertainty_path = self._uncertainty_output_path / input_image_file_path.name
        if not self._uncertainty_output_path.exists():
            self._uncertainty_output_path.mkdir()
        SimpleITK.WriteImage(uncertainty_map, str(uncertainty_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "segmentation": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "uncertainty": [
                dict(type="metaio_image", filename=uncertainty_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:

        image = SimpleITK.GetArrayFromImage(input_image)
        image = np.transpose(np.array(image))

        # The image must be normalized as that is what we did with monai for training of the model
        # only normalize non-zero values (i.e. not the background)
        non_zeros = image != 0
        mu = np.mean(image[non_zeros])
        sigma = np.std(image[non_zeros])
        image[non_zeros] = (image[non_zeros] - mu) / sigma

        with torch.no_grad():
            image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image).to(self.device), axis=0), axis=0)
            print(image.shape)

            outputs, confidences = sliding_window_inference(image, self.roi_size, self.sw_batch_size, self.super_model,
                                                            mode='gaussian')
            print(outputs.shape)
            print(confidences.shape)
            outputs = self.act(outputs)
            outputs = torch.argmax(outputs, axis=1).cpu().numpy()
            outputs = np.squeeze(outputs[0, 1])
            print(outputs.shape)


        outputs = np.squeeze(outputs)
        print(outputs.shape)
        conf_map = confidences.cpu().numpy()
        outputs = remove_connected_components(outputs)
        print(outputs.shape)
        uncs = - conf_map
        print(uncs.shape)
        out_seg = SimpleITK.GetImageFromArray(outputs)
        out_unc = SimpleITK.GetImageFromArray(uncs)
        return out_seg, out_unc


if __name__ == "__main__":
    Baseline().process()
