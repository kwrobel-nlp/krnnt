from setuptools import setup

setup(name='krnnt',
      version='0.1.2',
      description='Part of speech tagger for Polish',
      url='http://github.com/djstrong/krnnt',
      author='Krzysztof Wróbel',
      author_email='kwrobel@agh.edu.pl',
      packages=['krnnt'],
      license='GPL',
      python_requires='>=3, <4',
      install_requires=[
          'tensorflow-gpu','h5py','Keras','numpy','progress','progressbar2','regex','requests','setproctitle','termcolor','jsonlines'
      ],
      zip_safe=False)
