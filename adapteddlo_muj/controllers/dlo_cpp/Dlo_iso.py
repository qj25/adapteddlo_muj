# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _Dlo_iso
else:
    import _Dlo_iso

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class DLO_iso(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, dim_np, dim_bf0, theta_n, overall_rot, a_bar, b_bar):
        _Dlo_iso.DLO_iso_swiginit(self, _Dlo_iso.new_DLO_iso(dim_np, dim_bf0, theta_n, overall_rot, a_bar, b_bar))
    nodes = property(_Dlo_iso.DLO_iso_nodes_get, _Dlo_iso.DLO_iso_nodes_set)
    overall_rot = property(_Dlo_iso.DLO_iso_overall_rot_get, _Dlo_iso.DLO_iso_overall_rot_set)
    d_vec = property(_Dlo_iso.DLO_iso_d_vec_get, _Dlo_iso.DLO_iso_d_vec_set)
    nv = property(_Dlo_iso.DLO_iso_nv_get, _Dlo_iso.DLO_iso_nv_set)
    edges = property(_Dlo_iso.DLO_iso_edges_get, _Dlo_iso.DLO_iso_edges_set)
    bigL_bar = property(_Dlo_iso.DLO_iso_bigL_bar_get, _Dlo_iso.DLO_iso_bigL_bar_set)
    bf0_bar = property(_Dlo_iso.DLO_iso_bf0_bar_get, _Dlo_iso.DLO_iso_bf0_bar_set)
    alpha_bar = property(_Dlo_iso.DLO_iso_alpha_bar_get, _Dlo_iso.DLO_iso_alpha_bar_set)
    beta_bar = property(_Dlo_iso.DLO_iso_beta_bar_get, _Dlo_iso.DLO_iso_beta_bar_set)
    j_rot = property(_Dlo_iso.DLO_iso_j_rot_get, _Dlo_iso.DLO_iso_j_rot_set)
    p_thetan = property(_Dlo_iso.DLO_iso_p_thetan_get, _Dlo_iso.DLO_iso_p_thetan_set)
    bf0mat = property(_Dlo_iso.DLO_iso_bf0mat_get, _Dlo_iso.DLO_iso_bf0mat_set)
    qe_o2m_loc = property(_Dlo_iso.DLO_iso_qe_o2m_loc_get, _Dlo_iso.DLO_iso_qe_o2m_loc_set)
    qe_m2o_loc = property(_Dlo_iso.DLO_iso_qe_m2o_loc_get, _Dlo_iso.DLO_iso_qe_m2o_loc_set)
    excl_joints = property(_Dlo_iso.DLO_iso_excl_joints_get, _Dlo_iso.DLO_iso_excl_joints_set)
    nintgsteps = property(_Dlo_iso.DLO_iso_nintgsteps_get, _Dlo_iso.DLO_iso_nintgsteps_set)
    step_const = property(_Dlo_iso.DLO_iso_step_const_get, _Dlo_iso.DLO_iso_step_const_set)
    step_gain = property(_Dlo_iso.DLO_iso_step_gain_get, _Dlo_iso.DLO_iso_step_gain_set)
    distmat = property(_Dlo_iso.DLO_iso_distmat_get, _Dlo_iso.DLO_iso_distmat_set)

    def updateVars(self, dim_np, dim_bf0, dim_bfe):
        return _Dlo_iso.DLO_iso_updateVars(self, dim_np, dim_bf0, dim_bfe)

    def calculateCenterlineF2(self, dim_nf):
        return _Dlo_iso.DLO_iso_calculateCenterlineF2(self, dim_nf)

    def calculateCenterlineTorq(self, dim_nt, dim_nq, excl_jnts):
        return _Dlo_iso.DLO_iso_calculateCenterlineTorq(self, dim_nt, dim_nq, excl_jnts)

    def calculateF2LocalTorq(self):
        return _Dlo_iso.DLO_iso_calculateF2LocalTorq(self)

    def updateTheta(self, theta_n):
        return _Dlo_iso.DLO_iso_updateTheta(self, theta_n)

    def resetTheta(self, theta_n, overall_rot):
        return _Dlo_iso.DLO_iso_resetTheta(self, theta_n, overall_rot)

    def changeAlphaBeta(self, a_bar, b_bar):
        return _Dlo_iso.DLO_iso_changeAlphaBeta(self, a_bar, b_bar)

    def initQe_o2m_loc(self, dim_qo2m):
        return _Dlo_iso.DLO_iso_initQe_o2m_loc(self, dim_qo2m)

    def calculateOf2Mf(self, dim_mato, dim_matres):
        return _Dlo_iso.DLO_iso_calculateOf2Mf(self, dim_mato, dim_matres)

    def angBtwn3(self, dim_v1, dim_v2, dim_va):
        return _Dlo_iso.DLO_iso_angBtwn3(self, dim_v1, dim_v2, dim_va)

    def calculateEnergy(self):
        return _Dlo_iso.DLO_iso_calculateEnergy(self)
    __swig_destroy__ = _Dlo_iso.delete_DLO_iso

# Register DLO_iso in _Dlo_iso:
_Dlo_iso.DLO_iso_swigregister(DLO_iso)



