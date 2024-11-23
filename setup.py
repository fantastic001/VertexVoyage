from setuptools import setup, find_packages

setup(
    name='vertex_voyage',
    version='0.1',
    description='Graph management in distributed environment',
    url='https://github.com/fantastic001/vertex_voyage',
    packages=find_packages(),
    install_requires=[
        'cdlib',
        'networkx',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'zookeeper',
        'kazoo',
        'binpacking',
        'httplib2',
        'lxml',
        'pyyaml',
        'gensim',
        'mpi4py',
        'tabulate'
    ],
    entry_points={
        'console_scripts': [
            'vertex_voyage = vertex_voyage.__main__:main',
            'client = client.__main__:main'
        ]
    }
)
