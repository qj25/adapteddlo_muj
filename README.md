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
cd ../xpbd_cpp
bash swigbuild.sh
cd ../geds_cpp
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
python speed_test.py --newstart 1 --models plain,native,xfrc,adapt,massspring,jpqder,xpbd,geds
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

Run from `scripts/` after `pip install -e .` and building the C++ backends (step 2).

**Full pipeline** (recommended): bending stiffness for testids 0–4, then twisting stiffness, per model:
```
cd scripts
python real2sim_paramiden_all.py --wirecolor white --models massspring,cosserat
```

Default models: `adapt`, `native`, `massspring`, `cosserat`. Available models are registered in `adapteddlo_muj/envs/real2sim_paramiden/registry.py` (`adapt`, `native`, `massspring`, `cosserat`, `xfrc`, `xpbd`, `geds`).

**All wire colors**:
```
python real2sim_paramiden_all.py --wirecolors white,black,red --models adapt,native,massspring,cosserat
```

**Bending only** (e.g. if twisting data is not ready yet):
```
python real2sim_paramiden_all.py --wirecolor white --skip-twisting
```

**Twisting only** (after all five bending pickles exist):
```
python real2sim_paramiden_all.py --wirecolor white --skip-bending
```

**Search ranges** (defaults are model-aware; bending and twisting use the same bounds):
- `adapt` / `native` / `xfrc`: `[0, 2]` on `stiff_scale` and `beta/alpha`
- `massspring` / `cosserat` / `xpbd` / `geds`: `[0, 20]` on `stiff_scale` and `beta/alpha`

Override globally:
```
python real2sim_paramiden_all.py --wirecolor white --models massspring --stiff-lim 0,30 --b-a-lim 0,30
```

**Single step** (one testtype and one testid):
```
python real2sim_paramiden.py --testtype bending --testid 0 --wirecolor white --models massspring
python real2sim_paramiden.py --testtype twisting --wirecolor white --models massspring
```

Outputs:
- Per-trial bending alpha: `adapteddlo_muj/data/dlo_muj_real/stiff_vals/{color}_{model}_{testid}_bendstiff.pickle`
- Final combined stiffness `[alpha, b_a]`: `adapteddlo_muj/data/dlo_muj_real/stiff_vals/{color}_{model}_stiff.pickle` (used by step 9)

9. To simulate DLO held by Denso VS-060 robot arm in 4 different poses (modular model runners):

Stiffness values are read from `adapteddlo_muj/data/dlo_muj_real/stiff_vals/` (produce these with step 8).

**Default** (all default models, all wire colors, all four move poses):
```
cd scripts
python test_shape_w_arm.py
```

**Available models** (registered in `adapteddlo_muj/envs/test_shape_w_arm/registry.py`):

| Model | Backend | Notes |
|-------|---------|--------|
| `adapt` | `ValidRnR2Env`, adapted DLO stiffness | Default |
| `native` | `ValidRnR2Env`, native MuJoCo cable stiffness | Default |
| `massspring` | `ValidRnR2Env`, mass-spring backend | Default; uses `{color}_adapt_stiff.pickle` if massspring stiff file is missing |
| `xfrc` | `ValidRnR2Env`, direct external-force formulation | |
| `geds` | `ValidRnR2Env`, GEDS backend | Requires `geds_cpp` build (step 2) |
| `jpqder` | `ValidRnR3Env`, MuJoCo plugin `wire` | Requires wire plugin; see [C++ plugin](#c++-plugin) |

Default models: `adapt`, `native`, `massspring`.

**Select models** (`--models` overrides `--stiff`):
```
python test_shape_w_arm.py --models adapt,native,massspring
python test_shape_w_arm.py --models jpqder
python test_shape_w_arm.py --models adapt,geds,jpqder
```

**Legacy single-model flag** (same names as `--models`):
```
python test_shape_w_arm.py --stiff native
```

**Filter wire color and move pose** (`moveid` 0–3):
```
python test_shape_w_arm.py --models adapt --wirecolor white --moveid 1
python test_shape_w_arm.py --models native --wirecolor red --moveid 2
```

**Visualization**:
```
python test_shape_w_arm.py --models adapt --wirecolor white --moveid 0 --render 1
```

Model implementations live under `adapteddlo_muj/envs/test_shape_w_arm/` (one module per model). To compare sim shapes against real data afterward, run step 10 (`simvreal_dlomuj.py`) with the same model names; plugin sim pickles for `jpqder` are expected under `adapteddlo_muj/data/simdata/plugin/` as `simdata_{color}{moveid}_jpqder.pickle` (legacy `*_adapt2.pickle` is also accepted).

# Compare:
10. To compare sim and real DLO poses (modular model runners):
```
python simvreal_dlomuj.py
```
Optional model selection:
```
python simvreal_dlomuj.py --models adapt,native,massspring,jpqder,xpbd,geds
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
   - shape test: `adapteddlo_muj/envs/test_shape_w_arm/<new_model>.py`
   - parameter identification: `adapteddlo_muj/envs/real2sim_paramiden/<new_model>.py`
2. Register it:
   - update `adapteddlo_muj/envs/speed_test/registry.py` and/or
   - update `adapteddlo_muj/envs/simvreal_test/registry.py` and/or
   - update `adapteddlo_muj/envs/test_shape_w_arm/registry.py` and/or
   - update `adapteddlo_muj/envs/real2sim_paramiden/registry.py`
3. Run with explicit model selection:
```
python speed_test.py --newstart 1 --models <new_model>
python simvreal_dlomuj.py --models <new_model>
python test_shape_w_arm.py --models <new_model>
python real2sim_paramiden_all.py --models <new_model> --wirecolor white
```
4. (Optional) Add it to `DEFAULT_MODELS` in each registry if you want it included by default.

Validate GEDS after building:

```bash
python adapteddlo_muj/controllers/geds_cpp/geds_validate.py
```

Note:
- adjust time step to ensure stability
