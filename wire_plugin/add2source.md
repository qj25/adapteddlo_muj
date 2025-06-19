# Adding your new composite to source code
Steps:
1. In mujoco/src/user/user_composite.cc, change 
`(plugin_name != "mujoco.elasticity.cable")` to `(plugin_name != "mujoco.elasticity.cable" && plugin_name != "mujoco.elasticity.wire_qst")` in
```
// check plugin compatibility
  // TODO: move mujoco.elasticity.cable to the engine
  if (plugin.active) {
    if (type != mjCOMPTYPE_CABLE) {
      return comperr(error, "Only cable composite supports plugins", error_sz);
    }
    if (plugin_name != "mujoco.elasticity.cable") {
      return comperr(error, "Only mujoco.elasticity.cable is supported by composites", error_sz);
    }
  }
```

then, change all `type == mjCOMPTYPE_CABLE` to `type == mjCOMPTYPE_CABLE || type == mjCOMPTYPE_WIREQST` and all `(type != mjCOMPTYPE_CABLE)` to `(type != mjCOMPTYPE_CABLE && type != mjCOMPTYPE_WIREQST)`.

next, add 
```
case mjCOMPTYPE_WIREQST:
    return MakeWire(model, body, error, error_sz);
```
below 
```
case mjCOMPTYPE_CABLE:
    return MakeCable(model, body, error, error_sz);
```

Finally, add the contents of addwire.cc below AddCableBody function in user_composite.cc. 

To the header file user_composite.h, add `bool MakeWire(mjCModel* model, mjsBody* body, char* error, int error_sz);` below `bool MakeCable(mjCModel* model, mjsBody* body, char* error, int error_sz);`, and `mjsBody* AddWireBody(mjCModel* model, mjsBody* body, int ix, double normal[3], double prev_quat[4]);` below `mjsBody* AddCableBody(mjCModel* model, mjsBody* body, int ix, double normal[3], double prev_quat[4]);`.


2. Add `mjCOMPTYPE_WIREQST` to mujoco/src/user/user_composite.h.
```
typedef enum _mjtCompType {
  mjCOMPTYPE_PARTICLE = 0,
  mjCOMPTYPE_GRID,
  mjCOMPTYPE_CABLE,
  mjCOMPTYPE_ROPE,
  mjCOMPTYPE_LOOP,
  mjCOMPTYPE_CLOTH,

  mjNCOMPTYPES
} mjtCompType;
```

3. In src/xml/xml_native_reader.cc, add `{"wire_qst",       mjCOMPTYPE_WIREQST}` to `mjMap comp_map`
```
// composite type
const mjMap comp_map[mjNCOMPTYPES] = {
  {"particle",    mjCOMPTYPE_PARTICLE},
  {"grid",        mjCOMPTYPE_GRID},
  {"rope",        mjCOMPTYPE_ROPE},
  {"loop",        mjCOMPTYPE_LOOP},
  {"cable",       mjCOMPTYPE_CABLE},
  {"cloth",       mjCOMPTYPE_CLOTH}
};
```

<!-- 1. In mujoco/src/user/user_composite.cc, add #include <algorithm> and 
std::vector<mjtcomptype> compusable = {mjCOMPTYPE_CABLE, mjCOMPTYPE_WIREQST};
bool type_usable = (std::find(compusable.begin(), compusable.end(), type) != compusable.end());
  -->
