
from setuptools import setup

setup(
   name='ARIMA',
   version='0.1',
   description='ARIMA model estimation and hypothesis testing',
   author='Krystian Zawistowski',
   author_email='krystian.zawistowski@zoho.com',
   packages=['arima'],  #same as name
   install_requires=['scipy>=1.0.0', 'numpy>=1.13.1'], #external packages as dependencies
)