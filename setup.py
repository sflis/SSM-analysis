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
    dependency_links=["https://github.com/cta-chec/SSDAQ/tarball/master#egg=ssdaq"],
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


print("Downloading Hipparcos star catalog...")
if not os.path.exists("ssm/resources"):
    os.makedirs("ssm/resources")

"https://www.dropbox.com/s/7on3kmh58sz9j6w/testpix_m.hkl?raw=1"
"https://www.dropbox.com/s/ydziqcia6j6xm19/HipparcosCatalog_lt9.txt?raw=1"
"ssm/resources/HipparcosCatalog_lt9.txt"

from urllib.error import URLError
from urllib.request import urlopen


def download_resource(url, path, name):
    attempts = 0
    success = False
    while attempts < 3:
        try:
            response = urlopen(url, timeout=5)
            content = response.read()
            f = open(path, "wb")
            f.write(content)
            f.close()
            success = True
            break
        except URLError as e:
            attempts += 1
            print(type(e))
    if not success:
        print(
            "Failed to download {}. \n".format(name)
            + "Try later to download it manually from {}".format(url)
            + "\n and put it in ssm/resources"
        )


download_resource(
    "https://www.dropbox.com/s/ydziqcia6j6xm19/HipparcosCatalog_lt9.txt?raw=1",
    "ssm/resources/HipparcosCatalog_lt9.txt",
    "HipparcosCatalog_lt9",
)
download_resource(
    "https://www.dropbox.com/s/7on3kmh58sz9j6w/testpix_m.hkl?raw=1",
    "ssm/resources/testpix_m.hkl",
    "test pixel response parametrization",
)
rmtree("tmps")
