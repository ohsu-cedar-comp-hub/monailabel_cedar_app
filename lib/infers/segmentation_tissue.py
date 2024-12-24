# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
from functools import partial
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Union, Tuple, List, Mapping

import cv2
import inference.vista_2d
import inference.vista_2d.model
import monai
import monai.data
import numpy as np
import skimage
from tifffile import TiffFile, imwrite
import yaml
import torch

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.tasks.infer.basic_infer import CallBackTypes
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    EnsureType,
    CastToTyped,
)
from monai.data.image_reader import PILReader, ITKReader
from torch.cuda.amp import autocast
from inference.utils.monai_utils import sliding_window_inference

import inference

logger = logging.getLogger(__name__)

# A flag to use mps under MacBook Pro -- added by GW
# TODO: push it into the configuration
use_mps = False
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'    

# Technically we should use BasicInferTask. However, just to quickly set up the
# workflow using the existing Python script, we will implement InferTask directly
# for the time being. 
class SegmentationTissueInferTask(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation model over prostate cancer tissues.
    """

    def __init__(
        self,
        path: Union[None, str, Sequence[str]],
        class_id_to_name_color: Mapping[int, Tuple[str, List[int]]],
        type: Union[str, InferType] = InferType.SEGMENTATION,
        labels: Union[str, None, Sequence[str], Dict[Any, Any]] = None,
        dimension: int = 2,
        description: str = "A pre-trained segmentation model for prostate cancer tissues",
        config: Union[None, Dict[str, Any]] = None,    
    ):
        super().__init__(
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            config=config)
        self.path = [] if not path else [path] if isinstance(path, str) else path
        self.model = None
        self.class_id_to_name_color = class_id_to_name_color
        parser = self._config_args()
        args = self._read_arg_values()
        logger.info('Customized arguments for the inferrer: {}'.format(args))
        self.args = parser.parse_args(args)
        # In case out_channels is changed
        self.configed_out_channels = self.args.out_channels
        

    def _config_args(self):
        # Use YAML to configure the running instead of using parameters
        # Load the YAML configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'cedar_segmentation_config.yml')
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        # Create the argument parser
        parser = argparse.ArgumentParser(description=config['config']['description'])
        # Add arguments from the YAML configuration
        for arg in config['config']['arguments']:
            kwargs = {k: v for k, v in arg.items() if k != 'name'}
            if 'type' in kwargs.keys():
                type_text = kwargs['type']
                if type_text == 'int':
                    kwargs['type'] = int
                elif type_text == 'str':
                    kwargs['type'] = str
                elif type_text == 'float':
                    kwargs['type'] = float
            parser.add_argument(arg['name'], **kwargs)
        return parser
    

    def _read_arg_values(self) -> List[str]:
        # Use YAML to configure the running instead of using parameters
        # Load the YAML configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config_values.yml')
        if not os.path.exists(config_file):
            return []
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        args = []
        # Add arguments from the YAML configuration
        for arg in config['config']['arguments']:
            args.append(arg['name'])
            if 'value' in arg.keys():
                # Make sure they are in string
                args.append(str(arg['value']))
        return args
    

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["prostate_tissue"] = True
        return d
    
    def config(self) -> Dict[str, Any]:
        return self._config

    # This function is directly copied from basic_infer.py
    def get_path(self, validate=True):
        if not self.path:
            return None
        paths = self.path
        for path in reversed(paths):
            if path:
                if not validate or os.path.exists(path):
                    return path
        return None

    def is_valid(self) -> bool:
        # Make sure all models exist
        for path in self.path:
            if not os.path.exists(path):
                logger.error('model not existing: {}'.format(path))
                return False
        return True

    def __call__(
        self, request, callbacks: Union[Dict[CallBackTypes, Any], None] = None
    ) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        logger.debug('calling inference inside __call__: {}'.format(request))
        
        src_image_dir = request.get('src_image_dir')
        src_image_file = request.get("src_image_file")
        annotation_dir = request.get('annotation_dir')
        if annotation_dir is None or len(annotation_dir) == 0:
            annotation_dir = src_image_dir

        # For background and foreground segmentation
        out_channels = request.get('out_channels')
        if out_channels is None: # Use the default
            out_channels = self.configed_out_channels
        else:
            out_channels = int(out_channels) # Need to use int
        if self.args.out_channels != out_channels: # Need to reset the out_channels
            self.args.out_channels = out_channels
            model_dir = os.path.dirname(self.path[0])
            if out_channels == 2: # Need to use the segmentation only model
                self.path = [os.path.join(model_dir, 'res1024_model_best_fgbg.pt')]
            else:
                self.path = [os.path.join(model_dir, '1024res_model_best_081624.pt')]
            logger.info('Reset the model to {} for out_channels {}'.format(self.path, out_channels))
            self.model = None # reset so that we can reload the model

        test_loader = self.build_dataloader(src_image_dir, src_image_file)
        model = self.get_model()
        masked_image = self.infer(test_loader, model)
        
        json_path = self.get_annotation_file(annotation_dir, src_image_file)
        annotation_json = self.map_to_class(masked_image, json_path)
        annotation_json["label_names"] = self.labels
        # Apparent the first two are required by the RESTful API
        annotation_json["latencies"] = {}
        annotation_json['result_file'] = json_path
        # The first is the annnotation file and the second is some configuration file
        # Apparently in this setting, we have to return these two results. Otherwise, it cannot
        # work!
        return json_path, annotation_json
    

    def get_annotation_file(self, src_image_dir, src_image_file) -> Path:
        image = os.path.splitext(src_image_file)[0]
        return Path(src_image_dir, image + ".geojson")


    def map_to_class(self, multiclass_mask, save_path: Union[str, Path] = None):
        # For some reason, we have to make a copy here, which is not a good use of memory
        binary_mask = copy.deepcopy(multiclass_mask)
        binary_mask[multiclass_mask > 0] = 255

        binary_mask = skimage.morphology.remove_small_objects(binary_mask.astype(bool), min_size=2000)
        binary_mask = skimage.morphology.remove_small_holes(binary_mask, area_threshold=500000)
        binary_mask = binary_mask.astype("uint8")*255

        contours = skimage.measure.find_contours(binary_mask, 0.5, fully_connected="high")
        contours = [skimage.measure.approximate_polygon(c, tolerance = 2) for c in contours]

        geo_json = {"type": "FeatureCollection", "features":[]}

        for c in contours:

            # make sure first and last point are the same
            if np.any(c[0] != c[-1]): 
                c = c.tolist()
                c += [c[0]]
                c = np.array(c)

            rr, cc = skimage.draw.polygon(c[:,0], c[:,1]) # get pixel coords for mask
            label = np.argmax(np.bincount(multiclass_mask[rr, cc])) # find most frequent class for assignment
            
            feature = {
                "type":"Feature",
                "geometry":{
                    "type": "Polygon",
                    # Why is flip needed here?
                    "coordinates": [np.flip(c, axis=1).tolist()]
                    },
                "properties":{
                    "objectType":"annotation",
                    # Avoid to output empty content
                    # "name":"",
                    "classification":{
                        # Just a hack to use two channels for segmentation only
                        "name": self.class_id_to_name_color[label][0] if self.args.out_channels > 2 else 'unclassified',
                        # Cannot extract number in QuPath. Therefore, move it
                        # to metadata as a hack
                        # "number": str(label),
                        "color": self.class_id_to_name_color[label][1]
                        },
                    "metadata":{
                        # "ANNOTATION_DESCRIPTION": "",
                        "anno_style": "auto",
                        # Need to use string for JSON serialization
                        'class_id': str(label) if self.args.out_channels > 2 else '-1' # Use -1 as a flag for manual editing
                        }
                    }     
                }
            geo_json["features"].append(feature)

        # Better to let the QuPath app to handle the text instead of here to avoid any conflict.
        # if self.args.save_infer and save_path is not None:
        #     with open(save_path, "w") as outfile:
        #         json.dump(geo_json, outfile)
        #         logger.info('Saved annotations: {}'.format(save_path))
        return geo_json
    

    def infer(self, test_loader, model) -> np.ndarray:
        """There should be only one image file in test_loader. The inferred results are returned
        as np.ndarray for directly process.
        """
        if use_mps:
            device = 'mps'
        else:
            device = f'cuda:{self.args.rank}'

        # TODO: Check if this is needed together with y_pred
        post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.0, dtype=torch.uint8)])
        # There should be only one image file in the test_loader. 
        # We will return the masked image file name.
        masked_image = None
        with torch.no_grad():
            for _, batch_data in enumerate(test_loader):
                # A hard fix
                if batch_data['image'].shape[1] == 4:
                    batch_data['image'] = batch_data['image'][:, :3, :, :]

                if self.args.infer_only:
                    labels_l = None
                else:
                    # only take 1 batch
                    labels_l = batch_data["label"].as_subclass(torch.Tensor)[:, :1, ...]
                    # remove some rare labels (16, 17, 18, 19)
                    mapping_index = labels_l >= self.args.out_channels
                    if mapping_index.any():
                        labels_l[mapping_index] = 0

                file_name = batch_data["image"].meta['filename_or_obj'][0].split("/")[-1].split(".")[0]

                _device_in = "cpu"
                _device_out = "cpu"

                if use_mps:
                    _device_in = "mps"
                    _device_out = "mps"

                with autocast(enabled=True):
                    val_outputs = None
                    torch.cuda.empty_cache()
                    val_outputs = sliding_window_inference(
                        inputs=batch_data["image"].as_subclass(torch.Tensor).half().to(_device_in),
                        roi_size=[1024, 1024],
                        sw_batch_size=1,
                        predictor=model,
                        mode="gaussian",
                        overlap=0.25,
                        sw_device=device,
                        device=_device_out,
                        labels=labels_l.to(_device_in) if labels_l is not None else labels_l,
                        progress=True,
                        val_point_sampler=partial(self.prepare_sam_test_input, args=self.args)
                    )

                # Need to check if this is useful in this context
                # y_pred = torch.stack(post_pred(monai.data.utils.decollate_batch(val_outputs)))

                if self.args.save_infer:
                    # Ensure val_outputs is on GPU
                    # Define masking tensor only once to save memory
                    # Ensure val_outputs is on GPU
                    
                    # Step 1: Set the first channel to zero in-place, mask the remaining channels above the threshold of 0.5
                    val_outputs[:, 0] = 0  # Set the first channel to zero
                    val_outputs[:, 1:] = (val_outputs[:, 1:] > 0.5).half() * val_outputs[:, 1:]  # Threshold and mask other channels in-place

                    # Step 2: Perform argmax on GPU to get the output mask
                    save_np = torch.argmax(val_outputs, dim=1).squeeze().cpu().numpy().astype(np.uint8)  # Move to CPU for saving

                    # Step 3: Transpose and save as image
                    save_np = np.transpose(save_np)
                    masked_img_file = os.path.join(self.args.logdir, file_name + "_pred.tif")
                    cv2.imwrite(masked_img_file, save_np)
                    logger.info('Saved masked image: {}'.format(masked_img_file))

                    # Final result for further processing (still on CPU as numpy)
                    masked_image = save_np

                    # CPU version
                    # val_outputs = val_outputs.cpu() * torch.cat([torch.zeros(1, 1, *val_outputs.shape[-2:]),
                    #                                              torch.ones(1, val_outputs.shape[1]-1, *val_outputs.shape[-2:])], dim=1)
                    # val_outputs = (val_outputs > 0.5).half() * val_outputs
                    # save_np = np.transpose(torch.argmax(val_outputs, dim=1).squeeze().cpu().numpy().astype(np.uint8))
                    # masked_img_file = os.path.join(self.args.logdir, file_name+"_pred.tif")
                    # cv2.imwrite(masked_img_file, save_np)
                    # logger.info('Saved masked image: {}'.format(masked_img_file))
                    # #TODO: Need to check if val_outputs can be used by the class mapping
                    # masked_image = save_np
        return masked_image


    def handle_ome_tiff(self,
                        src_image_path: Path) -> Path:
        """Dump a tiff image file first from the ome.tiff file. It would be nicer to add an image reader
           to handle this action instead of this hacking way.
        Args:
            src_image_path (Path): the source image in the format ome.tiff

        Returns:
            Path: the path to the generated image
        """
        logger.info('Exporting the largest image in {}...'.format(src_image_path))
        # Get the path
        file_name = src_image_path.name.replace(".ome.tif", ".tif")
        output_path = Path(self.args.logdir, file_name)
        with TiffFile(src_image_path) as tif:
            # Access image series within the OME-TIFF
            images = [page.asarray() for page in tif.pages]

            # Find the largest image by area (width * height)
            largest_image = max(images, key=lambda img: img.shape[1] * img.shape[2])
            # Dump the largest image
            imwrite(output_path, largest_image)
            return output_path


    def build_dataloader(self,
                         src_image_dir: str,
                         src_image_file: str) -> monai.data.DataLoader:
        logger.info('Processing the image file: {}/{}'.format(src_image_dir, src_image_file))

        src_image_path = Path(src_image_dir, src_image_file)
        if src_image_file.endswith('.ome.tif') or src_image_file.endswith('.ome.tiff'):
            src_image_path = self.handle_ome_tiff(src_image_path)

        test_files = [{'image': str(src_image_path)}]
        
        keys = ["image"]
        val_transforms = Compose([
            # Need to use this reader for old tiff files
            LoadImaged(keys=keys, reader=PILReader, image_only=True),
            # LoadImaged(keys=keys, reader=ITKReader, image_only=True),
            EnsureChannelFirstd(keys=keys),
            CastToTyped(keys=keys, dtype=[torch.uint8] if self.args.infer_only else [torch.uint8, torch.uint8]),
        ])

        test_ds = monai.data.Dataset(
            data=test_files, transform=val_transforms
        )
        test_loader = monai.data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        return test_loader

    
    # Just be lazy. Need to specify the returned type!
    def get_model(self) -> any:
        if self.model is not None:
            return self.model
        model = inference.vista_2d.model.sam_model_registry[self.args.sam_base_model](checkpoint=None,
                                                    image_size=self.args.sam_image_size,
                                                    encoder_in_chans=self.args.roi_z_iter * 3,
                                                    patch_embed_3d=self.args.patch_embed_3d,
                                                    enable_auto_branch=self.args.enable_auto_branch
                                                    )
        if use_mps:
            device = 'mps'
        else:
            device = f'cuda:{self.args.rank}'

        model_dict = torch.load(self.get_path(), map_location=device)
        if "state_dict" in model_dict.keys():
            model_dict = model_dict["state_dict"]

        model.load_state_dict(model_dict, strict=True)
        
        if use_mps:
            model.to(torch.device('mps'))
        else:
            model.cuda(self.args.gpu)
        model.eval()
        # Cache the model so that we don't need to reload
        self.model = model
        return model


    def prepare_sam_test_input(self, inputs, labels, args, previous_pred=None):
        if use_mps:
            unique_labels = torch.tensor([i for i in range(0, args.out_channels)], device=torch.device('mps'))
        else:
            unique_labels = torch.tensor([i for i in range(0, args.out_channels)]).cuda(args.rank)
            # unique_labels = torch.tensor([i for i in range(0, args.out_channels)], device=torch.device('cpu'))

        if labels is not None:
            # preprocess make the size of lable same as high_res_logit
            batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()
        else:
            batch_labels = torch.zeros(1)

        prepared_input = [{"image": inputs, "original_size": tuple(inputs.shape)[1:]}]
        if args.label_prompt:
            labels_prompt = unique_labels.unsqueeze(-1)
            prepared_input[0].update(
                {"labels": labels_prompt})

        # if args.point_prompt:
        #     # need labels to simulate user's click when doing interactive inference
        #     assert labels is not None
        #     point_coords, point_labels = generate_point_prompt(batch_labels, args, points_pos=args.points_val_pos,
        #                                                     points_neg=args.points_val_neg, previous_pred=previous_pred)
        #     prepared_input[0].update(
        #         {"point_coords": point_coords, "point_labels": point_labels})

        if use_mps:
            return prepared_input, batch_labels.unsqueeze(1).to(torch.device('mps')), unique_labels
        else:
            return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), unique_labels
            # return prepared_input, batch_labels.unsqueeze(1).to(torch.device('cpu')), unique_labels
