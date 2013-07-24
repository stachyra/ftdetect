#!/user/bin/env python

from distutils.core import setup
import os

shortdesc = 'An assortment of algorithms related to computer vision and ' + \
            'image feature detection' 

readme = os.path.join(os.path.dirname(__file__), 'ftdetect', 'README.rst')

setup(
    name='ftdetect',
    version='1.0.0',
    author='Andrew L. Stachyra',
    author_email='andrewlstachyra@gmail.com',
    description=shortdesc,
    long_description=open(readme).read(),
    packages=['ftdetect'],
    package_dir={'ftdetect': 'ftdetect'},
    package_data={'ftdetect': ['images\\blocksTest.gif',
                               'images\\house.png',
                               'README.rst']}
    )
    
