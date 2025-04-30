This Python-based app provides inference and fine-tuning capabilities for [MiroSCOPE](https://github.com/ohsu-cedar-comp-hub/qupath_monaillabel_plugin), an AI-driven digital pathology platform for annotating functional tissue units (FTUs) based on QuPath.

It is developed based on the [MONAI Label custom app guide](https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/README.md#creating-a-custom-app)

<br />

### Installation

1. Follow the [MONAI Label Installation Guide](https://docs.monai.io/projects/label/en/latest/installation.html) to install all required packages.

2. Navigate to the `cedar_nvidia_pilot/monai_label_app` folder and install the package with:  
   ```bash
   pip install -e .
   ```
   **Note**: The `monai_label_app` code is merged from the [CEDAR-NVIDIA-Pilot GitHub Repository](https://github.com/ohsu-cedar-comp-hub/CEDAR-NVIDIA-Pilot/tree/monai_label_app)

<br />

### Running Inference in MiroSCOPE

#### Start the Monai Label App

1. On **macOS** or **Linux**, run:  
   ```bash
   run_monailabel_cedar_model.sh

2. On **Windows, run:
   ```command
   run_monailabel_cedar_model.bat
   ```
   

#### Configure model settings in MiroSCOPE

1. Download a model checkpoint and corresponding classes file from [Synapse](https://www.synapse.org/Synapse:syn66304443)

   Checkpoints on Synapse:

   | Checkpoint File             | Description                              | Corresponsing `classes.txt` File |
   |-----------------------------|------------------------------------------|----------------------------------|
   | `Fg-bg2_prostate.pt`        | Prostate segmentation only               | `prostate_cancer_classes.txt`    |
   | `multiclass_prostate.pt`    | Prostate segmentation + classification   | `prostate_cancer_classes.txt`    |
   | `Fg-bg_breast_pilot.pt`     | Breast segmentation                      | `breast_cancer_classes.txt`      |
   | `multiclass_breast_pilot.pt`| Breast segmentation + classification     | `breast_cancer_classes.txt`      |

   Or finetune a custom checkpoint, see section for Fine-tuning the Model

2. While the MiroSCOPE build of QuPath is running, navigate to `Edit` > `Preferences` > `MiroSCOPE`

3. Move the downloaded checkpoint and class files to the working directory, and paste their paths in the **FTU ID Class File** and **Model File** boxes respectively

4. If the model is a Fg-bg segmentation only version, check the **Use Inference Model for Segmentation Only** box


#### Run Inference

   Once the MONAI Label app is running and the model settings are configured, the **Infer Annotation** button in the MiroSCOPE tab of the QuPath app will intiate inference on the current image.

<br />

### Fine-tuning the Model

To fine-tune the model for use in MiroSCOPE, or to run inference outside of the platform, follow the instructions in the [inference_pipeline](./cedar_nvidia_pilot/inference_pipeline/) folder.
