from setuptools import setup, find_packages

setup(
    name='primp-python',
    version='0.2.0',
    packages=find_packages(),
    url='https://chirikjianlab.github.io/primp-page/',
    license='BSD-3 Clause',
    author='Sipu Ruan',
    author_email='',
    description='PRobabilistically-Informed Motion Primitives (PRIMP) for learning from demonstration',

    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'finitediff',
        'dtw-python',
        'movement_primitives',
        'roboticstoolbox-python'        
    ]
)
