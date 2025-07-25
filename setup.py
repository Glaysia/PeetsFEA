# setup.py
import setuptools
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="PeetsFEA",                                      # PyPI package name
    version="0.1.7",                                      # release version
    author="Glaysia",                                     # author name
    author_email="willbecat27@gmail.com",                 # author email
    description="전력전자 자동화에 필요한 유틸리티 모듈 모음",  # short description
    # README.md as long description
    long_description=long_description,
    long_description_content_type="text/markdown",        # format of long_description
    url="https://github.com/Glaysia/PeetsFEA",            # project URL

    # only package discovery here
    packages=setuptools.find_packages(),
    # map package name to folder
    package_dir={"peetsfea": "peetsfea"},

    # supported Python versions
    python_requires=">=3.10",
    install_requires=[                                    # dependencies with platform markers
        "pyaedt==0.17.4",                                 # AEDT Python API
        "pycaret==3.3.2",                                 # data science library
        "ipykernel>=6.30.0",                              # Jupyter kernel for Python
        "torch==2.7.1",                                   # PyTorch for machine learning
        "seaborn==0.13.2",                                # statistical data visualization
    ],

    classifiers=[                                         # PyPI metadata
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    # include files from MANIFEST.in
    include_package_data=True,
)
