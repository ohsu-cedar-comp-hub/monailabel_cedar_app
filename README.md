This Python-based app provides inference and fine-tuning capabilities for [MiroSCOPE](https://github.com/ohsu-cedar-comp-hub/qupath_monaillabel_plugin/tree/main?tab=readme-ov-file), an AI-driven digital pathology platform for annotating functional tissue units (FTUs) based on QuPath.

It is developed based on the MONAI Label custom app guide:  
https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/README.md#creating-a-custom-app

### Installation

1. Follow the [MONAI Label Installation Guide](https://docs.monai.io/projects/label/en/latest/installation.html) to install all required packages.

2. This app requires code from the **CEDAR-NVIDIA-Pilot** repository (use the `monai_label_app` branch):  
   [CEDAR-NVIDIA-Pilot GitHub Repository](https://github.com/ohsu-cedar-comp-hub/CEDAR-NVIDIA-Pilot/tree/monai_label_app)  

   **Note:** The required code has already been copied into the `cedar_nvidia_pilot/monai_label_app` folder in this project. Use the local copy provided here.

3. Navigate to the `cedar_nvidia_pilot/monai_label_app` folder and install the package with:  
   ```bash
   pip install -e .
Test
### Start the App

1. On **macOS** or **Linux**, run:  
   ```bash
   run_monailabel_cedar_model.sh

2. On **Windows, run:
   ```command
   run_monailabel_cedar_model.bat

### Fine-tuning the Model

To fine-tune the model for use in MiroSCOPE, follow the instructions in the [inference_pipeline](./cedar_nvidia_pilot/inference_pipeline/) folder.