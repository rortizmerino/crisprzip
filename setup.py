from setuptools import setup

setup(
    name='crisprzip-model',
    version='0.0.0',
    packages=['czmodel'],
    install_requires=['numpy==1.20.3', 'scipy==1.7.1', 'joblib==1.1.0',
                      'matplotlib==3.6.2', 'numba==0.56.4']
)
