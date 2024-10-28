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
import ctypes.util
import logging
import os
import platform
from ctypes import cdll
from typing import Dict

import monailabel
from monailabel.datastore.dsa import DSADatastore
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import strtobool

import lib.configs

logger = logging.getLogger(__name__)

# For windows (preload openslide dll using file_library) https://github.com/openslide/openslide-python/pull/151
if platform.system() == "Windows":
    # lib_path = str(ctypes.util.find_library("libopenslide-0.dll"))
    # if lib_path is None:
    # Just need to use the absolute path to point the dll lib
    lib_path = 'C:\\openslide-bin-4.0.0.5-windows-x64\\bin\\libopenslide-1.dll'
    print('libopenslide-0.dll path: {}'.format(lib_path))
    cdll.LoadLibrary(lib_path)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")
        logger.info('model_dir: {}'.format(self.model_dir))
        
        configs = {}
        candidates = get_class_names(lib.configs, "TaskConfig")
        for c in candidates:
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}

        models = conf.get("models", "all")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, None)

        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"CEDAR - Prostate Tisse Segementation",
            description="DeepLearning models for prostate cancer tissues",
            version='0.0.1',
        )

    def init_remote_datastore(self) -> Datastore:
        """
        -s http://0.0.0.0:8080/api/v1
        -c dsa_folder 621e94e2b6881a7a4bef5170
        -c dsa_api_key OJDE9hjuOIS6R8oEqhnVYHUpRpk18NfJABMt36dJ
        -c dsa_asset_store_path /localhome/sachi/Projects/digital_slide_archive/devops/dsa/assetstore
        """

        logger.info(f"Using DSA Server: {self.studies}")
        folder = self.conf.get("dsa_folder")
        annotation_groups = self.conf.get("dsa_groups", None)
        api_key = self.conf.get("dsa_api_key")
        asset_store_path = self.conf.get("dsa_asset_store_path")

        return DSADatastore(
            api_url=self.studies,
            api_key=api_key,
            folder=folder,
            annotation_groups=annotation_groups,
            asset_store_path=asset_store_path,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v

        #################################################
        # Pipeline based on existing infers
        #################################################
        # if infers.get("nuclick") and infers.get("classification_nuclei"):
        #     p = infers["nuclick"]
        #     c = infers["classification_nuclei"]
        #     if isinstance(p, NuClick) and isinstance(c, BasicInferTask):
        #         p.init_classification(c)

        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers

        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t
        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            # "wsi_random": WSIRandom(),
        }

        if strtobool(self.conf.get("skip_strategies", "false")):
            return strategies

        for n, task_config in self.models.items():
            s = task_config.strategy()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Strategy:: {k} => {v}")
                strategies[k] = v

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies

