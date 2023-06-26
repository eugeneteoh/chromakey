from setuptools import find_packages, setup
from pathlib import Path

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="chromakey",
    version="0.1.1",
    packages=find_packages(),
    install_requires=["Pillow", "scipy", "numpy"],
    url="https://github.com/eugeneteoh/chromakey",
    author="Eugene Teoh",
    author_email="eugenetwc1@gmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type='text/markdown' 
)
