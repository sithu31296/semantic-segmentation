from setuptools import setup, find_packages

setup(
    name='semseg',
    version='0.4.1',
    description='SOTA Semantic Segmentation Models',
    url='https://github.com/sithu31296/semantic-segmentation',
    author='Sithu Aung',
    author_email='sithu31296@gmail.com',
    license='MIT',
    packages=find_packages(include=['semseg']),
    install_requires=[
        'tqdm',
        'tabulate',
        'numpy',
        'scipy',
        'matplotlib',
        'tensorboard',
        'fvcore',
        'einops',
        'rich',
    ]
)