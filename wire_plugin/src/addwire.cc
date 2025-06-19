bool mjCComposite::MakeWire(mjCModel* model, mjsBody* body, char* error, int error_sz) {
  // check dim
  if (dim != 1) {
    return comperr(error, "Wire must be one-dimensional", error_sz);
  }

  // check geom type
  if (def[0].spec.geom->type != mjGEOM_CYLINDER &&
      def[0].spec.geom->type != mjGEOM_CAPSULE &&
      def[0].spec.geom->type != mjGEOM_BOX) {
    return comperr(error, "Wire geom type must be sphere, capsule or box", error_sz);
  }

  // add name to model
  mjsText* pte = mjs_addText(&model->spec);
  mjs_setString(pte->name, ("composite_" + prefix).c_str());
  mjs_setString(pte->data, ("wire_" + prefix).c_str());

  // populate uservert if not specified
  if (uservert.empty()) {
    for (int ix=0; ix < count[0]; ix++) {
      double v[3];
      for (int k=0; k < 3; k++) {
        switch (curve[k]) {
          case mjCOMPSHAPE_LINE:
            v[k] = ix*size[0]/(count[0]-1);
            break;
          case mjCOMPSHAPE_COS:
            v[k] = size[1]*cos(mjPI*ix*size[2]/(count[0]-1));
            break;
          case mjCOMPSHAPE_SIN:
            v[k] = size[1]*sin(mjPI*ix*size[2]/(count[0]-1));
            break;
          case mjCOMPSHAPE_ZERO:
            v[k] = 0;
            break;
          default:
            // SHOULD NOT OCCUR
            mju_error("Invalid composite shape: %d", curve[k]);
            break;
        }
      }
      mjuu_rotVecQuat(v, v, quat);
      uservert.insert(uservert.end(), v, v+3);
    }
  }

  // create frame
  double normal[3], prev_quat[4];
  mjuu_setvec(normal, 0, 1, 0);
  mjuu_setvec(prev_quat, 1, 0, 0, 0);

  // add one body after the other
  for (int ix=0; ix < count[0]-1; ix++) {
    body = AddWireBody(model, body, ix, normal, prev_quat);
  }

  // add skin
  if (def[0].spec.geom->type == mjGEOM_BOX) {
    if (skinsubgrid > 0) {
      count[1]+=2;
      MakeSkin2Subgrid(model, 2*def[0].spec.geom->size[2]);
      count[1]-=2;
    } else {
      count[1]++;
      MakeSkin2(model, 2*def[0].spec.geom->size[2]);
      count[1]--;
    }
  }
  return true;
}

mjsBody* mjCComposite::AddWireBody(mjCModel* model, mjsBody* body, int ix,
                                    double normal[3], double prev_quat[4]) {
  char txt_geom[100], txt_site[100], txt_slide[100];
  char this_body[100], next_body[100], this_joint[100];
  double dquat[4], this_quat[4];

  // set flags
  int lastidx = count[0]-2;
  bool first = ix == 0;
  bool last = ix == lastidx;
  bool secondlast = ix == lastidx-1;

  // compute edge and tangent vectors
  double edge[3], tprev[3], tnext[3], length_prev = 0;
  mjuu_setvec(edge, uservert[3*(ix+1)+0]-uservert[3*ix+0],
                    uservert[3*(ix+1)+1]-uservert[3*ix+1],
                    uservert[3*(ix+1)+2]-uservert[3*ix+2]);
  if (!first) {
    mjuu_setvec(tprev, uservert[3*ix+0]-uservert[3*(ix-1)+0],
                       uservert[3*ix+1]-uservert[3*(ix-1)+1],
                       uservert[3*ix+2]-uservert[3*(ix-1)+2]);
    length_prev = mjuu_normvec(tprev, 3);
  }
  if (!last) {
    mjuu_setvec(tnext, uservert[3*(ix+2)+0]-uservert[3*(ix+1)+0],
                       uservert[3*(ix+2)+1]-uservert[3*(ix+1)+1],
                       uservert[3*(ix+2)+2]-uservert[3*(ix+1)+2]);
    mjuu_normvec(tnext, 3);
  }

  // update moving frame
  double length = mjuu_updateFrame(this_quat, normal, edge, tprev, tnext, first);

  // create body, joint, and geom names
  if (first) {
    mju::sprintf_arr(this_body, "%sB_first", prefix.c_str());
    mju::sprintf_arr(next_body, "%sB_%d", prefix.c_str(), ix+1);
    mju::sprintf_arr(this_joint, "%sJ_first", prefix.c_str());
    mju::sprintf_arr(txt_site, "%sS_first", prefix.c_str());
  } else if (last) {
    mju::sprintf_arr(this_body, "%sB_last", prefix.c_str());
    mju::sprintf_arr(next_body, "%sB_last2", prefix.c_str());  // Changed to point to B_last2
    mju::sprintf_arr(this_joint, "%sJ_last", prefix.c_str());
    mju::sprintf_arr(txt_site, "%sS_last", prefix.c_str());
  } else if (secondlast){
    mju::sprintf_arr(this_body, "%sB_%d", prefix.c_str(), ix);
    mju::sprintf_arr(next_body, "%sB_last", prefix.c_str());
    mju::sprintf_arr(this_joint, "%sJ_%d", prefix.c_str(), ix);
  } else {
    mju::sprintf_arr(this_body, "%sB_%d", prefix.c_str(), ix);
    mju::sprintf_arr(next_body, "%sB_%d", prefix.c_str(), ix+1);
    mju::sprintf_arr(this_joint, "%sJ_%d", prefix.c_str(), ix);
  }
  mju::sprintf_arr(txt_geom, "%sG%d", prefix.c_str(), ix);
  mju::sprintf_arr(txt_slide, "%sJs%d", prefix.c_str(), ix);

  // add body
  body = mjs_addBody(body, 0);
  mjs_setString(body->name, this_body);
  if (first) {
    mjuu_setvec(body->pos, offset[0]+uservert[3*ix],
                           offset[1]+uservert[3*ix+1],
                           offset[2]+uservert[3*ix+2]);
    mjuu_copyvec(body->quat, this_quat, 4);
    if (frame) {
      mjs_setFrame(body->element, frame);
    }
  } else {
    mjuu_setvec(body->pos, length_prev, 0, 0);
    double negquat[4] = {prev_quat[0], -prev_quat[1], -prev_quat[2], -prev_quat[3]};
    mjuu_mulquat(dquat, negquat, this_quat);
    mjuu_copyvec(body->quat, dquat, 4);
  }

  // add geom
  mjsGeom* geom = mjs_addGeom(body, &def[0].spec);
  mjs_setDefault(geom->element, mjs_getDefault(body->element));
  mjs_setString(geom->name, txt_geom);
  if (def[0].spec.geom->type == mjGEOM_CYLINDER ||
      def[0].spec.geom->type == mjGEOM_CAPSULE) {
    mjuu_zerovec(geom->fromto, 6);
    geom->fromto[3] = length;
  } else if (def[0].spec.geom->type == mjGEOM_BOX) {
    mjuu_zerovec(geom->pos, 3);
    geom->pos[0] = length/2;
    geom->size[0] = length/2;
  }

  // add plugin
  if (plugin.active) {
    mjsPlugin* pplugin = &body->plugin;
    pplugin->active = true;
    pplugin->element = plugin.element;
    mjs_setString(pplugin->plugin_name, mjs_getString(plugin.plugin_name));
    mjs_setString(pplugin->name, plugin_instance_name.c_str());
  }

  // update orientation
  mjuu_copyvec(prev_quat, this_quat, 4);

  // add curvature joint
  if (!first || strcmp(initial.c_str(), "none")) {
    mjsJoint* jnt = mjs_addJoint(body, &defjoint[mjCOMPKIND_JOINT][0].spec);
    mjs_setDefault(jnt->element, mjs_getDefault(body->element));
    jnt->type = (first && strcmp(initial.c_str(), "free") == 0) ? mjJNT_FREE : mjJNT_BALL;
    jnt->damping = jnt->type == mjJNT_FREE ? 0 : jnt->damping;
    jnt->armature = jnt->type == mjJNT_FREE ? 0 : jnt->armature;
    jnt->frictionloss = jnt->type == mjJNT_FREE ? 0 : jnt->frictionloss;
    mjs_setString(jnt->name, this_joint);
  }

  // exclude contact pair
  if (!last) {
    mjsExclude* exclude = mjs_addExclude(&model->spec);
    mjs_setString(exclude->bodyname1, std::string(this_body).c_str());
    mjs_setString(exclude->bodyname2, std::string(next_body).c_str());
  }

  // add site at the boundary
  if (last || first) {
    mjsSite* site = mjs_addSite(body, &def[0].spec);
    mjs_setDefault(site->element, mjs_getDefault(body->element));
    mjs_setString(site->name, txt_site);
    mjuu_setvec(site->pos, last ? length : 0, 0, 0);
    mjuu_setvec(site->quat, 1, 0, 0, 0);
  }

  // If this is the last body, add the additional B_last2 body
  if (last) {
    // Create B_last2 body
    mjsBody* last2_body = mjs_addBody(body, 0);
    char last2_name[100];
    mju::sprintf_arr(last2_name, "%sB_last2", prefix.c_str());
    mjs_setString(last2_body->name, last2_name);
    
    // Set position and orientation
    mjuu_setvec(last2_body->pos, length, 0, 0);
    mjuu_copyvec(last2_body->quat, this_quat, 4);

    // Add plugin to B_last2 if active
    if (plugin.active) {
      mjsPlugin* last2_plugin = &last2_body->plugin;
      last2_plugin->active = true;
      last2_plugin->element = plugin.element;
      mjs_setString(last2_plugin->plugin_name, mjs_getString(plugin.plugin_name));
      mjs_setString(last2_plugin->name, plugin_instance_name.c_str());
    }
  }

  return body;
} 