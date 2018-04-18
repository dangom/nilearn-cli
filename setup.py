from io import open

from setuptools import find_packages, setup

with open('nilearn_cli/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.rst', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = ['nilearn', 'numpy', 'scipy', 'scikit-learn']

setup(
    name='nilearn-cli',
    version=version,
    description='Convenient command line tools for nilearn',
    long_description=readme,
    author='Daniel Gomez',
    author_email='d.gomez@donders.ru.nl',
    maintainer='Daniel Gomez',
    maintainer_email='d.gomez@donders.ru.nl',
    url='https://github.com/dangom/nilearn-cli',
    license='MIT/Apache-2.0',

    keywords=[
        'nilearn',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    entry_points={
        'console_scripts': [
            'connectome = nilearn_cli.connectome:run_connectome',
        ]
    },
    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
)
