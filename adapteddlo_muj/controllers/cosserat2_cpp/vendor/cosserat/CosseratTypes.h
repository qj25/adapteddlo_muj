#pragma once

#include <glm/glm.hpp>
#include <limits>
#include <vector>

#include "../kitten/Rotor.h"

#define COSSERAT_THREAD_BATCH (90)

namespace Cosserat {
using namespace glm;
using namespace Kitten;

struct Seg {
    Rotor q;
    ivec2 i;
    float kss;
    float l;
};

struct Vert {
    vec3 pos;
    float invMass;
};

struct RestAngle {
    Rotor qRest;
    ivec2 i;
    float kbt;
};

struct MetaData {
    float h = 1 / 120.f;
    vec3 gravity = vec3(0, -9.8f, 0);
    float damping = 1e-7f;
    float drag = 0.1f;
    int useProportionalDamping = 1;
    int numItr = 4;
    float lastH = 1 / 120.f;
    float time = 0;
};

class Sim {
public:
    std::vector<Seg> segs;
    std::vector<Vert> verts;
    std::vector<RestAngle> restAngles;

    std::vector<vec3> vels;
    std::vector<vec3> dxs;

    std::vector<int> vertSegIndices;
    std::vector<int> vertSegIds;
    std::vector<int> segAngleIndices;
    std::vector<int> segAngleIds;

    MetaData meta;
    bool multithreaded = false;

    void init();
    void addVertex(vec3 pos);
    void addSegment(ivec2 i, float kss = 1e2f, float density = 1.f);
    void addRestAngle(ivec2 i, float kbt = 1.f);
    void pinVertex(int i);
    void pinSegment(int i);

    void iterateVertVBD();
    void iterateSegLambda();
    void iterateStableCosserat();

    void copyFrom(const Sim& other);
};

}  // namespace Cosserat
