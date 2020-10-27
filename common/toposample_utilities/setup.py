"""
NEUTRINO - NEUral TRIbe and Network Observer
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import setup, find_packages


setup(
    name='toposample',
    version='0.10',
    install_requires=['h5py', 'numpy', 'simplejson', 'scipy>=1.0.0', 'scikit-learn', 'pandas',
                      'progressbar'],
    packages=find_packages(),
    include_package_data=True,
    author=['Michael Reimann','Henri Riihimäki','Jason Smith','Janis Lazovskis'],
    author_email=['michael.reimann@epfl.ch','henri.riihimaki@abdn.ac.uk','jason.smith@ntu.ac.uk','jlazovskis@gmail.com'],
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
