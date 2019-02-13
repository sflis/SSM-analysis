from setuptools import setup, find_packages

import sys
install_requires = ["zmq","numpy",'tables','pyparsing','matplotlib','pyyaml','click','nsb','SSDAQ','CHECLabPy']

version = {}
with open('ssm/version.py') as fp: 
    exec(fp.read(),version)

setup(name="ssm",
      version=version['__version__'],
      description="A framework to analyze and monitor slow signal data from the CHEC-S camera",
      author="Samuel Flis",
      author_email="samuel.flis@desy.de",
      url='https://github.com/sflis/SSM-analysis',
      packages=find_packages(),
      provides=["ssm"],
      license="GNU Lesser General Public License v3 or later",
      install_requires=install_requires,
      extras_requires={
          #'encryption': ['cryptography']
      },
      classifiers=["Programming Language :: Python",
                   "Programming Language :: Python :: 3",
                   "Development Status :: 4 - Beta",
                   "Intended Audience :: Developers",
                   "Operating System :: OS Independent",
                   "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
                   "Topic :: Software Development :: Libraries :: Python Modules",
                   ],
      entry_points={'console_scripts':
                    [
                        # 'ssdaq = ssdaq.bin.ssdaqd:main',
                    ]
                    }
      )