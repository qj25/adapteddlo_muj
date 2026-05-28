# Adapted Discrete Elastic Rod model for MuJoCo
An adaptation of the Cartesian stiffness forces in Discrete Elastic Rods ([Bergou2008](http://www.cs.columbia.edu/cg/pdfs/143-rods.pdf)) into generalized coordinates for use in a joint-based simulator, [MuJoCo](https://mujoco.readthedocs.io/en/latest/overview.html).

## C++ plugin
To use with C++ API of MuJoCo, see folder 'wire_plugin' for instructions.

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
2. Build C++ model backends:
```
cd adapteddlo_muj/controllers/dlo_cpp
bash swigbuild.sh
cd ../massspring_cpp
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
5. For computation speed tests (modular model runners):
```
python speed_test.py --newstart 1
```
Optional model selection:
```
python speed_test.py --newstart 1 --models plain,native,xfrc,adapt,massspring,jpq_der
```
Load and plot previously saved JSON results:
```
python speed_test.py --newstart 0 --models plain,native
```
Outputs are written per model to:
`adapteddlo_muj/data/speed_test/<test_type>_<model>.json`

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
python real2sim_paramiden.py
```
9. To simulate DLO held by Denso VS-060 robot arm in 4 different poses:
```
python test_shape_w_arm.py
```
# Compare:
10. To compare sim and real DLO poses (modular model runners):
```
python simvreal_dlomuj.py
```
Optional model selection:
```
python simvreal_dlomuj.py --models adapt,native,massspring
```
Optional filtering by wire color and move id:
```
python simvreal_dlomuj.py --models adapt --wirecolor white --moveid 1
```
Outputs are written per model to:
`adapteddlo_muj/data/simvreal_test/simvreal_<model>.json`

## Adding a new model to modular runners
1. Add a new model module:
   - speed test: `adapteddlo_muj/envs/speed_test/<new_model>.py`
   - simvreal test: `adapteddlo_muj/envs/simvreal_test/<new_model>.py`
2. Register it:
   - update `adapteddlo_muj/envs/speed_test/registry.py` and/or
   - update `adapteddlo_muj/envs/simvreal_test/registry.py`
3. Run with explicit model selection:
```
python speed_test.py --newstart 1 --models <new_model>
python simvreal_dlomuj.py --models <new_model>
```
4. (Optional) Add it to `DEFAULT_MODELS` in each registry if you want it included by default.

Note:
- adjust time step to ensure stability
