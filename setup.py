from distutils.core import setup
setup(
  name     = 'open_deployment',
  packages = ['open_deployment'],
  version  = '0.0.1',
  license  = 'MIT',
  description  = 'Open Deployment helps you deploy your ML models by seamlesslly creating APIs, dockerfiles and much more. ',   # Give a short description about your library
  author       = 'Larxel',
  author_email = 'andrew.mvd@gmail.com',
  url          = 'https://github.com/user/reponame',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Larxel/open_deployment/archive/0.0.1.tar.gz',    # I explain this later on
  keywords     = ['deployment', 'cloud', 'api'],
  install_requires = [],
  classifiers      = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)