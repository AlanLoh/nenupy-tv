from setuptools import setup, find_packages
import re
import nenupytv

meta_file = open('nenupytv/metadata.py').read()
metadata  = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", meta_file))

setup(
    name = 'nenupytv',
    packages = find_packages(),
    include_package_data = True,
    install_requires = ['numpy', 'astropy'],
    python_requires = '>=3.5',
    scripts = ['bin/nenufartv', 'bin/nenufartv_mp'],
    version = nenupytv.__version__,
    description = 'NenuFAR-TV Python package',
    url = 'https://github.com/AlanLoh/nenupy-tv.git',
    author = metadata['author'],
    author_email = metadata['email'],
    license = 'MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    zip_safe = False
    )

# make the package:
# python3 setup.py sdist bdist_wheel
# upload it:
# python3 -m twine upload dist/*version*

# Release:
# git tag -a v*version* -m "annotation for this release"
# git push origin --tags


# Install on nancep
# /usr/local/bin/pip3.5 install nenupytv --prefix=/cep/lofar/nenupytv --upgrade
