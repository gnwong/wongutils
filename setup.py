from setuptools import setup, find_packages

setup(name="wongutils",
      version="0.1.11",
      description="",
      url="https://github.com/gnwong/wongutils",
      author="gnwong",
      author_email="gnwong@ias.edu",
      license="MIT",
      packages=find_packages(),
      install_requires=["numpy", "scipy", "h5py", "tqdm"],
      zip_safe=False)
