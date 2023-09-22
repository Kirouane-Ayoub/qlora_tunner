from setuptools import setup, find_packages

setup(
    name="qlora_tunner",
    version="1.0",
    author="Kirouane Ayoub",
    author_email="ayoubkirouane3@email.com",
    description="A Python package for Qlora fine-tuning open-source Large language models",
    packages=find_packages(),
    install_requires=[
        "accelerate==0.21.0","peft==0.4.0"
        ,"bitsandbytes==0.40.2" ,"transformers==4.31.0"
        ,"trl==0.4.7" , "xformers" , "datasets"
    ],
)
