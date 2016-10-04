from setuptools import setup
from setuptools import find_packages

install_requires = [
    'numpy',
    'fuel'
]

setup(name='discgen',
      version='0.2.0',
      description='Discriminative Regularization for Generative Models',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/discgen',
      download_url='https://github.com/dribnet/discgen/archive/0.2.0.tar.gz',
      license='MIT',
      entry_points={
          # 'console_scripts': ['neupup = neupup.neupup:main']
      },
      install_requires=install_requires,
      packages=find_packages(exclude=['experiments', 'scripts', 'legacy']))
