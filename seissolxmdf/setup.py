import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seissolxmdf", 
    version="0.0.2",
    author="SeisSol Group",
    description="A python reader for SeisSol xdmf output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SeisSol/Visualization/seissolxmdf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
