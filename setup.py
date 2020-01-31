#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

from distutils.core import setup
from setuptools import find_packages
import os
this = os.path.dirname(__file__)

with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]

packages = find_packages(where='.', exclude=())
assert packages

# read version from the package file.
version_str = '1.0.0'
with (open(os.path.join(this, 'easyai/__init__.py'), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

README = os.path.join(os.getcwd(), "README.md")
with open(README) as f:
    long_description = f.read()

setup(
    name='easyai',
    version=version_str,
    description="Develop deep learning networks is easy",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT License',
    author='lipeijie',
    author_email='1014153254@qq.com',
    url='https://www.baidu.com/',
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: MIT License'],
)
