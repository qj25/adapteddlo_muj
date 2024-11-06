# Adapted Discrete Elastic Rod model for MuJoCo
An adaptation of the Cartesian stiffness forces in Discrete Elastic Rods ([Bergou2008](http://www.cs.columbia.edu/cg/pdfs/143-rods.pdf)) into generalized coordinates for use in a joint-based simulator, [MuJoCo](https://mujoco.readthedocs.io/en/latest/overview.html).

## Requires
On Ubuntu, make sure you have the packages

liblapack-dev
libopenblas-dev
installed:

> sudo apt install liblapack-dev libopenblas-dev

Also requires:
-[MuJoCo](https://github.com/google-deepmind/mujoco), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer), [eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)

## Use:
1. In the root directory of this package:
```
pip install -e .
```
2. Build dlo_cpp:
```
cd adapteddlo_muj/controllers/dlo_cpp
bash swigbuild.sh
cd ../../..
```
3. To obtain validation results:
```
cd scripts
python dlo_testdata.py
```
and change the settings in the scripts to your desired test: 'lhb' - localized helical buckling test, 'mbi' - Michell's buckling instability test.
4. For MBI overall results:
```
python plot_mbicombined.py
```
5. For computation speed tests:
```
python speed_test.py
```

Note:
- adjust time step to ensure stability
