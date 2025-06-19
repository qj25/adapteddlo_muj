# muj_addplugins
Additional Elasticity Plugins for MuJoCo v3.3.2. Required to build MuJoCo from source. wire_qst is the implementation of Discrete Elastic Rods ([Bergou2008](http://www.cs.columbia.edu/cg/pdfs/143-rods.pdf)) in generalized coordinates. This implementation can also be used with the python bindings built from source.

## Adding the plugins
1. Move desired .cc and .h files from the source folder to 'mujoco/plugin/elasticity'.
2. Register the plugin based on register.cc in the source folder.
3. Update the CMakeLists.txt based on the one in the source folder (including eigen3).
4. Add misc changes to source code as in add2source.md.
5. Include the plugin by following the instructions in how2use.md.
6. Wire model can now be used by including wire_qst.xml in your .xml file.