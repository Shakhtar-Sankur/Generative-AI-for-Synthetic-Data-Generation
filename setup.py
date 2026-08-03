from setuptools import setup, find_packages

setup(
    name="synthetic_medical_imaging",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1",
        "torchvision==0.15.2",
        "diffusers==0.21.0",
        "google-cloud-bigquery==3.11.0",
        "torch-fidelity==0.3.0",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
    ],
    author="Sankur Kundu",
    author_email="sankur.kundu.tw@gmail.com",
    description="Synthetic X-ray generation for rare lung diseases using DDPM",
    url="https://github.com/SankurTW/Synthetic-Medical-Imaging",
)