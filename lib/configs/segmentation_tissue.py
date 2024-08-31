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

from cProfile import label
import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, Union

import pandas as pd

from lib.infers.segmentation_tissue import SegmentationTissueInferTask

from monai.networks.nets import BasicUNet

from lib.infers import SegmentationTissueInferTask

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others import label_colors
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class SegmentationTissue(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Load the label configuration from an external file
        # This file should be the same used in the qupath cedar extension
        # and masks converting
        current_dir = os.path.dirname(os.path.abspath(__file__))
        class_file = os.path.join(current_dir, 'prostate_cancer_path_classes.txt')
        class_df = pd.read_csv(class_file, sep='\t')
        self.labels = dict(zip(class_df['class_name'], class_df['class_id']))
        self.label_colors = dict(zip(class_df['class_name'], class_df['color']))
        # Make sure color is an int array
        for name, color in self.label_colors.items():
            color_array = [int(c) for c in color.split(',')]
            self.label_colors[name] = color_array

        # Create a map from class id to name and color
        self.class_id_to_name_color = {class_id: (class_name, self.label_colors[class_name]) 
                                       for class_name, class_id in self.labels.items()}

        # Model Files
        self.path = [
            os.path.join(self.model_dir, '1024res_model_best_081624.pt')
        ]

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        # task: InferTask = SegmentationTissueInferTask()
        # return task
        # preload = strtobool(self.conf.get("preload", "false"))
        # roi_size = json.loads(self.conf.get("roi_size", "[1024, 1024]"))
        # logger.info(f"Using Preload: {preload}; ROI Size: {roi_size}")
        task: InferTask = SegmentationTissueInferTask(
            labels=self.labels,
            class_id_to_name_color=self.class_id_to_name_color,
            path=self.path # Specifiy for the model path
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        """ Disable training for the time being

        Returns:
            Optional[TrainTask]: _description_
        """
        return None
