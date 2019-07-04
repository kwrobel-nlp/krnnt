from setuptools import setup

setup(name='krnnt',
      version='0.2.0',
      description='Part of speech tagger for Polish',
      url='http://github.com/djstrong/krnnt',
      author='Krzysztof Wróbel',
      author_email='Wrobel.Krzysztof@gmail.com',
      packages=['krnnt'],
      license='LGPL',
      python_requires='>=3, <4',
      install_requires=[
          'tensorflow-gpu','h5py','Keras','numpy','progress','progressbar2','regex','requests','setproctitle','termcolor','jsonlines',
            'pytest','tqdm','flask','scikit-learn','gunicorn','gevent',
            'pytest-shell @ https://api.github.com/repos/djstrong/pytest-shell/tarball/'
      ],
      zip_safe=False)
#pytest, tqdm, flask, sklearn, git+https://github.com/djstrong/pytest-shell.git
#gunicorn gevent
