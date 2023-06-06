from setuptools import setup, find_packages

setup(
    name='primp-python',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='',
    author='Sipu Ruan',
    author_email='ruansp@nus.edu.sg',
    description='PRobabilistically-Informed Motion Primitives (PRIMP) for learning from demonstration',

    install_requires=[
        'roboticstoolbox-python',
        'movement_primitives',
        'finitediff',
        'numpy',
        'scipy',
        'matplotlib',
        'dtw-python'
    ]
)
