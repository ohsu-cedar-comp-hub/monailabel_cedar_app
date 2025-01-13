The app for cedar-nvidia-pilot. It is developed based on https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/README.md#creating-a-custom-app.

### Installation

1. Follow the instructions provided by **Monai Label**: [MONAI Label Installation Guide](https://docs.monai.io/projects/label/en/latest/installation.html) to install all required packages.  
   
2. This app requires some code from the **CEDAR-NVIDIA-Pilot** repository. Clone the repo using the following link:  
   [CEDAR-NVIDIA-Pilot GitHub Repository](https://github.com/ohsu-cedar-comp-hub/CEDAR-NVIDIA-Pilot/tree/monai_label_app)  
   **Note:** Make sure to use the `monai_label_app` branch.  
   
3. After cloning the repo, navigate to the project folder using `cd`, and then install the package with:  
   ```bash
   pip install -e .

### Start the App

1. On **macOS** or **Linux**, run:  
   ```bash
   run_monailabel_cedar_model.sh

2. On **Windows, run:
   ```command
   run_monailabel_cedar_model.bat

