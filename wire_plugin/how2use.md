## Build MuJoCo from Source
Instructions from the original [documentation](https://mujoco.readthedocs.io/en/latest/programming/#building-from-source) put here for convenience.
To build MuJoCo from source, you will need CMake and a working C++17 compiler installed. The steps are:

Clone the mujoco repository: `git clone https://github.com/deepmind/mujoco.git` then `git checkout 3.3.2`.

Create a new build directory and cd into it.
```
mkdir build
cd build
```
Configure the build, then build.
```
cmake ..
cmake --build .
```
### Python Bindings
(Note: requires python>3.9)
Enter python directory with `cd mujoco/python`.

Generate a source distribution tarball `bash make_sdist.sh`.

Use the generated source distribution to build and install the bindings
```
cd dist
export MUJOCO_PATH=/PATH/TO/MUJOCO \
export MUJOCO_PLUGIN_PATH=/PATH/TO/MUJOCO_PLUGIN \
pip install mujoco-x.y.z.tar.gz
```

## Running and Compiling
Run `export MJPLUGIN_PATH=$HOME/mujoco/build/lib/` in bash or add it to ~/.bashrc for future use. Scripts will use this to locate and loadPluginLibrary.

Use the following function to include the plugin in your code for use in your environment.
Python:
```
    def _init_plugins(self):
        plugin_path = os.environ.get("MJPLUGIN_PATH")
        if plugin_path:
            plugin_file = os.path.join(plugin_path, "libelasticity.so")
            try:
                mujoco.mj_loadPluginLibrary(plugin_file)
            except Exception as e:
                print(f"Failed to load plugin: {e}")
        else:
            print("MJPLUGIN_PATH is not set.")
```
C++:
```
    void load_plugins() {
        const char* plugin_path = std::getenv("MJPLUGIN_PATH");
        if (plugin_path) {
            std::string full_path = std::string(plugin_path) + "/libelasticity.so";
            mj_loadPluginLibrary(full_path.c_str());
        } else {
            std::cerr << "MJPLUGIN_PATH not set. Set it to the directory containing libelasticity.so." << std::endl;
        }
    }
```

## Notes
1. add B to the line after A in cmake/MujocoOptions.cmake and simulate/cmake/SimulateOptions.cmake
A: option(MUJOCO_BUILD_MACOS_FRAMEWORKS "Build libraries as macOS Frameworks" OFF)
B: option(MUJOCO_INSTALL_PLUGINS "Install MuJoCo plugins" ON)

2. add the following when using lglfw and lglew 
`-ldl -lGL -lglfw -L/usr/lib/x86_64-linux-gnu -l:libGLEW.so -pthread`
 
3. use `-I$HOME/mujoco/include -L$HOME/mujoco/build/lib` when compiling for mujoco/mujoco.h. g++ compiler does not recognize the ~ in ~/mujoco.
