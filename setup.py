from setuptools import setup

setup(name='krnnt',
      version='1.0.0',
      description='Part of speech tagger for Polish',
      url='https://github.com/kwrobel-nlp/krnnt',
      author='Krzysztof Wróbel',
      author_email='Wrobel.Krzysztof@gmail.com',
      packages=['krnnt'],
      license='LGPL',
      python_requires='>=3, <4',
      install_requires=[
          'Cython', 'h5py', 'Keras==2.2.5', 'numpy', 'regex', 'requests', 'jsonlines', 'tqdm', 'flask', 'gunicorn',
          'krnnt_utils @ git+https://github.com/Zhylkaaa/krnnt_text_utils@cython'
      ],
      extras_require={
          'train': ['scikit-learn'],
          'pytest': ['pytest', 'pytest-benchmark',
                     'pytest-shell @ https://api.github.com/repos/djstrong/pytest-shell/tarball/'],
          'tfcpu': ['tensorflow==1.14.0'],
          'tfgpu': ['tensorflow-gpu==2.5.1']
      },
      zip_safe=False)