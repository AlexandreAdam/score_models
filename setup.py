from setuptools import setup, find_packages

# Read the contents of the README file
with open("long_description.rst", "r") as fh:
    long_description = fh.read()

setup(
	name="score_models",
	version="0.4.6",
    description="A simple pytorch interface for score model and basic diffusion.",
    long_description=long_description,
    author="Alexandre Adam",
    author_email="alexandre.adam@umontreal.ca",
    url="https://github.com/AlexandreAdam/torch_score_models",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "scipy",
        "torch_ema",
        "h5py",
        "numpy",
        "tqdm"
    ],
	python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)

