from setuptools import setup, find_packages
import os
import sys
from shutil import copyfile, rmtree

install_requires = [
    "zmq",
    "numpy",
    "tables",
    "pyparsing",
    "matplotlib",
    "pyyaml",
    "click",
    "nsb",
    "SSDAQ",
    "CHECLabPy",
    "astropy",
    "pynverse",
    "ctapipe",
    "hickle",
]


if not os.path.exists("tmps"):
    os.makedirs("tmps")
copyfile("ssm/version.py", "tmps/version.py")
__import__("tmps.version")
package = sys.modules["tmps"]
package.version.update_release_version("ssm")


setup(
    name="ssm",
    version=package.version.get_version(pep440=True),
    description="A framework to analyze and monitor slow signal data from the CHEC-S camera",
    author="Samuel Flis",
    author_email="samuel.flis@desy.de",
    url="https://github.com/sflis/SSM-analysis",
    packages=find_packages(),
    provides=["ssm"],
    license="GNU Lesser General Public License v3 or later",
    install_requires=install_requires,
    dependency_links = ["https://github.com/cta-chec/SSDAQ.git#egg=ssdaq"],
    extras_requires={
        #'encryption': ['cryptography']
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            # 'ssdaq = ssdaq.bin.ssdaqd:main',
        ]
    },
)

rmtree("tmps")
