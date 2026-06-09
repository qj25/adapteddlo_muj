#include "CosseratTypes.h"

namespace Cosserat {

void Sim::iterateVertVBD() {
    const float h = meta.h;
    const float damping = meta.damping / h;
    const bool proportionalDamping = meta.useProportionalDamping != 0;

#ifdef _OPENMP
#pragma omp parallel for if(multithreaded) schedule(dynamic, COSSERAT_THREAD_BATCH)
#endif
    for (int tid = 0; tid < static_cast<int>(verts.size()); tid++) {
        auto v0 = verts[tid];

        if (v0.invMass > 0) {
            float H = 1 / (v0.invMass * h * h);
            vec3 dx = dxs[tid];
            vec3 f = (1 / (h * h * v0.invMass)) * (vels[tid] - dx);

            const int start = vertSegIndices[tid];
            const int end = vertSegIndices[tid + 1];
            for (int s = start; s < end; s++) {
                int segId = vertSegIds[s];
                auto seg = segs[segId];

                int oid = (seg.i.x == tid) ? seg.i.y : seg.i.x;
                auto p1 = verts[oid].pos;
                auto p1dx = dxs[oid];

                float invl = 1 / seg.l;
                float order = (seg.i.x == tid) ? 1 : -1;
                vec3 c = ((p1 - v0.pos) + (p1dx - dx)) * (order * invl) - seg.q * vec3(1, 0, 0);

                float k = std::abs(seg.kss) * invl;
                float d = k * invl;
                float damp = damping * (proportionalDamping ? d : invl);
                f += order * k * c - damp * dx;
                H += d + damp;
            }

            vec3 delta = f * (1 / H);
            dx += delta;
            dxs[tid] = dx;
        }
    }
}

}  // namespace Cosserat
