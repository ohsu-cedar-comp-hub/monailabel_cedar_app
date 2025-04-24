from setuptools import setup, find_packages

setup(
    name='cedar_nvidia_pilot',  # Replace with your project name
    version='0.0.1',          # Initial version
    packages=find_packages(),  # Automatically find packages in the directory
    description='This package is used to image segemenation and classification',  # Short description
    author='?????',        # Your name
    author_email='?????',  # Your email
    url='https://github.com/ohsu-cedar-comp-hub/CEDAR-NVIDIA-Pilot',  # URL to your project (if applicable)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change if using a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Minimum Python version required
)
