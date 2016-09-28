from setuptools import setup
from setuptools import find_packages

install_requires = [
    'numpy',
]

setup(name='discgen',
      version='0.1.0',
      description='Discriminative Regularization for Generative Models',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/discgen',
      download_url='https://github.com/dribnet/discgen/tarball/0.1.0',
      license='MIT',
      entry_points={
          # 'console_scripts': ['neupup = neupup.neupup:main']
      },
      install_requires=install_requires,
      packages=find_packages())
