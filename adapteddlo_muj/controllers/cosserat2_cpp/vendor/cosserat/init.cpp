#include "CosseratTypes.h"

#include <limits>
#include <stdexcept>

namespace Cosserat {

void Sim::addVertex(vec3 pos) {
    Vert v;
    v.pos = pos;
    v.invMass = std::numeric_limits<float>::infinity();
    vels.push_back(vec3(0));
    verts.push_back(v);
}

void Sim::addSegment(ivec2 i, float kss, float density) {
    if (i.x < 0 || i.x >= static_cast<int>(verts.size()) ||
        i.y < 0 || i.y >= static_cast<int>(verts.size())) {
        throw std::invalid_argument("Segment vertex index out of bounds");
    }

    Seg s;
    s.i = i;

    vec3 dir = verts[i.y].pos - verts[i.x].pos;
    s.l = length(dir);
    if (s.l <= 0) {
        throw std::invalid_argument("Invalid segment length");
    }

    s.q = Rotor::fromTo(vec3(1, 0, 0), dir / s.l);
    s.kss = kss * s.l;

    segs.push_back(s);

    const float m = 0.5f * density * s.l;
    if (verts[i.x].invMass > 0) {
        verts[i.x].invMass = 1 / (1 / verts[i.x].invMass + m);
    }
    if (verts[i.y].invMass > 0) {
        verts[i.y].invMass = 1 / (1 / verts[i.y].invMass + m);
    }
}

void Sim::addRestAngle(ivec2 i, float kbt) {
    if (i.x < 0 || i.x >= static_cast<int>(segs.size()) ||
        i.y < 0 || i.y >= static_cast<int>(segs.size())) {
        throw std::invalid_argument("Angle segment index out of bounds");
    }
    if (i.x == i.y) {
        throw std::invalid_argument("Angle segment indices are the same");
    }

    RestAngle ra;
    ra.i = i;
    ra.qRest = segs[i.x].q.inverse() * segs[i.y].q;
    ra.kbt = 4 * kbt / (0.5f * (segs[i.x].l + segs[i.y].l));
    restAngles.push_back(ra);
}

void Sim::pinVertex(int i) {
    if (i < 0 || i >= static_cast<int>(verts.size())) {
        throw std::invalid_argument("Vertex index out of bounds");
    }
    verts[i].invMass = 0;
}

void Sim::pinSegment(int i) {
    if (i < 0 || i >= static_cast<int>(segs.size())) {
        throw std::invalid_argument("Segment index out of bounds");
    }
    segs[i].kss = -std::abs(segs[i].kss);
}

void Sim::init() {
    std::vector<int> counts(verts.size() + 1, 0);
    for (auto& seg : segs) {
        counts[seg.i.x]++;
        counts[seg.i.y]++;
    }

    int runningSum = 0;
    vertSegIndices.resize(verts.size() + 1);
    for (size_t i = 0; i < vertSegIndices.size(); i++) {
        vertSegIndices[i] = runningSum;
        runningSum += counts[i];
        counts[i] = 0;
    }

    vertSegIds.resize(runningSum);
    for (size_t i = 0; i < segs.size(); i++) {
        ivec2 ind = segs[i].i;
        vertSegIds[vertSegIndices[ind.x] + counts[ind.x]++] = static_cast<int>(i);
        vertSegIds[vertSegIndices[ind.y] + counts[ind.y]++] = static_cast<int>(i);
    }

    counts.assign(segs.size() + 1, 0);
    for (auto& ra : restAngles) {
        counts[ra.i.x]++;
        counts[ra.i.y]++;
    }

    runningSum = 0;
    segAngleIndices.resize(segs.size() + 1);
    for (size_t i = 0; i < segAngleIndices.size(); i++) {
        segAngleIndices[i] = runningSum;
        runningSum += counts[i];
        counts[i] = 0;
    }

    segAngleIds.resize(runningSum);
    for (size_t i = 0; i < restAngles.size(); i++) {
        ivec2 ind = restAngles[i].i;
        segAngleIds[segAngleIndices[ind.x] + counts[ind.x]++] = static_cast<int>(i);
        segAngleIds[segAngleIndices[ind.y] + counts[ind.y]++] = static_cast<int>(i);
    }

    dxs.assign(verts.size(), vec3(0));
    vels.assign(verts.size(), vec3(0));

    if (segs.size() > COSSERAT_THREAD_BATCH * 2) {
        multithreaded = true;
    }
}

void Sim::copyFrom(const Sim& other) {
    segs = other.segs;
    verts = other.verts;
    restAngles = other.restAngles;
    vels = other.vels;
    dxs = other.dxs;
    vertSegIndices = other.vertSegIndices;
    vertSegIds = other.vertSegIds;
    segAngleIndices = other.segAngleIndices;
    segAngleIds = other.segAngleIds;
    meta = other.meta;
    multithreaded = other.multithreaded;
}

}  // namespace Cosserat
