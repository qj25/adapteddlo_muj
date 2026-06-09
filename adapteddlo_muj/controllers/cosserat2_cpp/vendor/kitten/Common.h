#pragma once

#define KITTEN_FUNC_DECL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/norm.hpp>

#include <cmath>
#include <cstdio>

namespace Kitten {
using namespace glm;

template <typename T>
KITTEN_FUNC_DECL inline T pow3(T v) {
    return v * v * v;
}

template <typename T>
KITTEN_FUNC_DECL inline mat<3, 3, T, defaultp> abT(vec<3, T, defaultp> a, vec<3, T, defaultp> b) {
    return mat<3, 3, T, defaultp>(
        b.x * a.x, b.y * a.x, b.z * a.x,
        b.x * a.y, b.y * a.y, b.z * a.y,
        b.x * a.z, b.y * a.z, b.z * a.z);
}

KITTEN_FUNC_DECL inline mat3 orthoBasisX(vec3 n) {
    mat3 basis;
    basis[0] = n;
    if (n.z >= n.y) {
        const float a = 1.0f / (1.0f + n.z);
        const float b = -n.x * n.y * a;
        basis[1] = vec3(b, 1.0f - n.y * n.y * a, -n.y);
        basis[2] = -vec3(1.0f - n.x * n.x * a, b, -n.x);
    } else {
        const float a = 1.0f / (1.0f + n.y);
        const float b = -n.x * n.z * a;
        basis[1] = vec3(1.0f - n.x * n.x * a, -n.x, b);
        basis[2] = -vec3(b, -n.z, 1.0f - n.z * n.z * a);
    }
    return basis;
}

template <typename T>
KITTEN_FUNC_DECL inline mat<3, 3, T, defaultp> crossMatrix(vec<3, T, defaultp> v) {
    return mat<3, 3, T, defaultp>(
        0, v.z, -v.y,
        -v.z, 0, v.x,
        v.y, -v.x, 0);
}

}  // namespace Kitten
