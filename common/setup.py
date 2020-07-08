from setuptools import setup, find_packages


setup(
    name='toposample',
    version='0.03',
    install_requires=['h5py', 'numpy', 'simplejson', 'scipy>=1.0.0', 'scikit-learn', 'pandas',
                      'progressbar'],
    packages=find_packages(),
    include_package_data=True,
    author='Michael Reimann',
    author_email='michael.reimann@epfl.ch',
    description='''Sample neuron communities according to topological metrics
    and evaluate their ability to decode stimuli''',
    license='LGPL-3.0',
    keywords=('neuroscience',
              'brain',
              'topology',
              'modelling'),
    url='http://bluebrain.epfl.ch',
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'License :: LGPL-3.0',
                 'Operating System :: POSIX',
                 'Topic :: Utilities',
                 ],
)
