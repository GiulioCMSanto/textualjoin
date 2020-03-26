from setuptools import setup

setup(author="Giulio Cesare Mastrocinque Santo",
      name='textualjoin',
      author_email="giuliosanto@gmail.com"
      version='v0.0.1',
      description='TextualJoin is a python package for join dataframes based on text data.',
      long_description=README,
      license="MIT",
      packages=setuptools.find_packages(),
      python_requires=">=3.7",
      zip_safe=False)