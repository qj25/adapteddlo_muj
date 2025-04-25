from setuptools import setup

setup(name='adapteddlo_muj',
      version='0.0.1',
      # packages=[
      #       package for package in find_packages() if package.startswith("rl_tut")
      # ]
      # python_requires='>3',
      install_requires=['gymnasium'], # And any other dependencies foo needs
      packages=['adapteddlo_muj'],
)

# install with command "pip install -e ."