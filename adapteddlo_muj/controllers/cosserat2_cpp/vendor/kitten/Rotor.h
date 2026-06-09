#pragma once

#include "Common.h"

namespace Kitten {
template <typename T>
struct RotorX {
    typedef glm::vec<3, T, glm::defaultp> q_type;
    typedef glm::vec<4, T, glm::defaultp> v_type;

    union {
        v_type v;
        struct {
            q_type q;
            T w;
        };
        struct {
            T x, y, z, s;
        };
    };

    KITTEN_FUNC_DECL static RotorX<T> angleAxis(T rad, q_type axis) {
        rad *= 0.5;
        return RotorX<T>(sin(rad) * axis, cos(rad));
    }

    KITTEN_FUNC_DECL static RotorX<T> fromTo(q_type from, q_type to) {
        q_type h = (from + to) * 0.5f;
        T l = length2(h);
        if (l > 0) h *= inversesqrt(l);
        else h = orthoBasisX((vec3)from)[1];

        return RotorX<T>(cross(from, h), dot(from, h));
    }

    KITTEN_FUNC_DECL static RotorX<T> identity() { return RotorX<T>(); }

    KITTEN_FUNC_DECL RotorX(T x, T y = 0, T z = 0, T w = 0) : v(x, y, z, w) {}
    KITTEN_FUNC_DECL RotorX(q_type q, T w = 0) : q(q), w(w) {}
    KITTEN_FUNC_DECL RotorX(v_type v) : v(v) {}
    KITTEN_FUNC_DECL RotorX() : v(0, 0, 0, 1) {}
    KITTEN_FUNC_DECL RotorX(const RotorX<T>& other) : v(other.v) {}

    KITTEN_FUNC_DECL RotorX<T> inverse() const { return RotorX<T>(-q, w); }

    KITTEN_FUNC_DECL q_type rotate(q_type v) const {
        q_type a = w * v + cross(q, v);
        T c = dot(v, q);
        return w * a + cross(q, a) + c * q;
    }

    KITTEN_FUNC_DECL mat<3, 3, T, defaultp> matrix() {
        mat3 cm = crossMatrix(q);
        return abT(q, q) + mat<3, 3, T, defaultp>(w * w) + 2 * w * cm + cm * cm;
    }

    KITTEN_FUNC_DECL static RotorX<T> fromMatrix(mat<3, 3, T, defaultp> m) {
        RotorX<T> q;
        T t;
        if (m[2][2] < 0) {
            if (m[0][0] > m[1][1]) {
                t = 1 + m[0][0] - m[1][1] - m[2][2];
                q = RotorX<T>(t, m[1][0] + m[0][1], m[0][2] + m[2][0], m[2][1] - m[1][2]);
            } else {
                t = 1 - m[0][0] + m[1][1] - m[2][2];
                q = RotorX<T>(m[1][0] + m[0][1], t, m[2][1] + m[1][2], m[0][2] - m[2][0]);
            }
        } else {
            if (m[0][0] < -m[1][1]) {
                t = 1 - m[0][0] - m[1][1] + m[2][2];
                q = RotorX<T>(m[0][2] + m[2][0], m[2][1] + m[1][2], t, m[1][0] - m[0][1]);
            } else {
                t = 1 + m[0][0] + m[1][1] + m[2][2];
                q = RotorX<T>(m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1], t);
            }
        }
        return RotorX<T>(((0.5f / glm::sqrt(t)) * q.v)).inverse();
    }

    KITTEN_FUNC_DECL friend q_type operator*(RotorX<T> lhs, const q_type& rhs) {
        return lhs.rotate(rhs);
    }

    KITTEN_FUNC_DECL friend RotorX<T> operator*(RotorX<T> lhs, const RotorX<T>& rhs) {
        return RotorX<T>(
            lhs.w * rhs.q + rhs.w * lhs.q + cross(lhs.q, rhs.q),
            lhs.w * rhs.w - dot(lhs.q, rhs.q));
    }

    KITTEN_FUNC_DECL RotorX<T>& operator+=(const RotorX<T>& rhs) {
        v += rhs.v;
        return *this;
    }

    KITTEN_FUNC_DECL friend RotorX<T> operator+(RotorX<T> lhs, const RotorX<T>& rhs) {
        return RotorX<T>(lhs.v + rhs.v);
    }

    KITTEN_FUNC_DECL q_type axis(T& angle) const {
        T l = length(q);
        if (l == 0) {
            angle = 0;
            return q_type(1, 0, 0);
        }
        angle = 2 * atan2(l, w);
        return q / l;
    }

    KITTEN_FUNC_DECL explicit operator v_type() const { return v; }

    KITTEN_FUNC_DECL T& operator[](std::size_t idx) { return v[idx]; }
};

template <typename T>
KITTEN_FUNC_DECL inline RotorX<T> normalize(RotorX<T> a) {
    return RotorX<T>(glm::normalize(a.v));
}

template <typename T>
KITTEN_FUNC_DECL inline RotorX<T> projectRotor(
    RotorX<T> x, vec<3, T, glm::defaultp> e, vec<3, T, glm::defaultp> d) {
    auto q = RotorX<T>::fromTo(e, d);
    auto qp = q.inverse() * x;
    qp.y = qp.z = 0;
    return q * normalize(qp);
}

using Rotor = RotorX<float>;
using RotorD = RotorX<double>;

}  // namespace Kitten
