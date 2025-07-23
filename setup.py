# setup.py
import setuptools
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="PeetsFEA",                                      # PyPI package name
    version="0.1.2",                                      # release version
    author="Glaysia",                                     # author name
    author_email="willbecat27@gmail.com",                 # author email
    description="전력전자 자동화에 필요한 유틸리티 모듈 모음",  # short description
    long_description=long_description,                    # README.md as long description
    long_description_content_type="text/markdown",        # format of long_description
    url="https://github.com/Glaysia/PeetsFEA",            # project URL

    packages=setuptools.find_packages(),                  # only package discovery here
    package_dir={"peetsfea": "peetsfea"},                 # map package name to folder

    python_requires=">=3.10",                             # supported Python versions
    install_requires=[                                    # dependencies with platform markers
        "pyaedt==0.17.4; sys_platform=='win32'",
        "duckdb==1.3.1; sys_platform=='win32'",
        "pyaedt==0.17.4; sys_platform=='linux'",
        "duckdb==1.3.1; sys_platform=='linux'",
    ],

    classifiers=[                                         # PyPI metadata
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    include_package_data=True,                            # include files from MANIFEST.in
)
