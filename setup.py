from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "scipy", "matplotlib"]

setup(
    name="nnoptic",
    version="1.0",
    author="Sergei Kuzmin",
    author_email="ssk251198@yandex.ru",
    description="A package to training and tuning neural network modeling linear optics scheme",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/SergeiKuzmin/nnoptic.git",
    packages=find_packages(),
    install_requires=requirements,
)
