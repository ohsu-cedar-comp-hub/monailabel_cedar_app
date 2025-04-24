The app for cedar-nvidia-pilot. It is developed based on https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/README.md#creating-a-custom-app.

### Installation

1. Follow the instructions provided by **Monai Label**: [MONAI Label Installation Guide](https://docs.monai.io/projects/label/en/latest/installation.html) to install all required packages.  
   
2. This app requires some code from the **CEDAR-NVIDIA-Pilot** repository (**note**: the monai_label_app branch):  
   [CEDAR-NVIDIA-Pilot GitHub Repository](https://github.com/ohsu-cedar-comp-hub/CEDAR-NVIDIA-Pilot/tree/monai_label_app)  
   
   **Note:** The code required from that repo has been copied into the cedar_nvidia_pilot/monai_label_app folder at this project. Use the code here instead.
   
3. `cd` to the cedar_nvidia_pilot/monai_label_app folder, and then install the package with:  
   ```bash
   pip install -e .

### Start the App

1. On **macOS** or **Linux**, run:  
   ```bash
   run_monailabel_cedar_model.sh

2. On **Windows, run:
   ```command
   run_monailabel_cedar_model.bat

