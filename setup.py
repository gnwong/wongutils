from pathlib import Path

from setuptools import setup, find_packages

setup(name="wongutils",
      version="0.1.11",
      description="Utilities for simulations, black hole calculations, and visualization",
      long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
      long_description_content_type="text/markdown",
      url="https://github.com/gnwong/wongutils",
      author="gnwong",
      author_email="gnwong@ias.edu",
      license="MIT",
      packages=find_packages(),
      install_requires=["numpy", "scipy", "h5py", "tqdm"],
      zip_safe=False)
