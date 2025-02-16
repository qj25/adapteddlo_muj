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
python dlo_testdata.py --stiff [stiff_type] --test [test_type]
```
where [stiff_type] = 'native' - native MuJoCo stiffness model, or 'adapt' - adapted DLO model,
and [test_type] = 'lhb' - localized helical buckling test, or 'mbi' - Michell's buckling instability test.
4. For MBI overall results:
```
python plot_mbicombined.py
```
5. For computation speed tests:
```
python speed_test.py
```
## Real experiments
# Real:
6. To obtain 2D shape from image for parameter identification:
```
python get_pos/dlomuj_2Dpos.py
```
7. To obtain depth from kinect azure (put csv file in data3d in .csv format -- x_pixelpos, y_pixelpos, pixeldepth):
```
python get_pos/get_depth_many_azure.py
```
# Sim:
8. To determine sim stiffness parameters from real experiment (with 2D positions and critical angles obtained):
```
python real_testdata.py
```
# Compare:
9. To compare sim and real wire poses:
```
python simvreal_dlomuj.py
```


Note:
- adjust time step to ensure stability
