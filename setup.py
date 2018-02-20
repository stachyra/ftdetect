from setuptools import setup
import os

shortdesc = 'An assortment of algorithms related to computer vision and ' + \
            'image feature detection' 

readme = os.path.join(os.path.dirname(__file__), 'README.rst')

setup(
    name='ftdetect',
    version='1.0.1',
    description=shortdesc,
    long_description=open(readme).read(),
    url='https://github.com/stachyra/ftdetect',
    author='Andrew L. Stachyra',
    author_email='andrewlstachyra@gmail.com',
    license='MIT',
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering :: Image Recognition'],
    packages=['ftdetect'],
    package_dir={'ftdetect': 'ftdetect'},
    install_requires=['numpy', 'scipy', 'matplotlib', 'Pillow'],
    package_data={'ftdetect': [os.path.join('images', 'blocksTest.gif'),
                               os.path.join('images', 'house.png'),
                               os.path.join('..', 'README.rst')]}
    )
