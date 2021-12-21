import setuptools
import re


def get_property(prop, project):
    # https://stackoverflow.com/questions/17791481/creating-a-version-attribute-for-python-packages-without-getting-into-troubl/41110107
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + "/__init__.py").read()
    )
    return result.group(1)


with open("README.md", "r") as fh:
    long_description = fh.read()

project_name = "seissolxdmfwriter"
setuptools.setup(
    name=project_name,
    version=get_property("__version__", project_name),
    author="SeisSol Group",
    description="A python writer for SeisSol xdmf output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SeisSol/Visualization/seissolxdmfwriter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "h5py"],
)
