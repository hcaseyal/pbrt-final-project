// shapes/fur.cpp*
#include "shapes/fur.h"
#include "paramset.h"
#include "stats.h"

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Curves", curveBytes);
STAT_PERCENT("Intersections/Ray-curve intersection tests", nHits, nTests);
STAT_INT_DISTRIBUTION("Intersections/Curve refinement level", refinementLevel);
STAT_COUNTER("Scene/Curves", nCurves);
STAT_COUNTER("Scene/Split curves", nSplitCurves);

// Fur Utility Functions
static Point3f BlossomBezier(const Point3f p[4], Float u0, Float u1, Float u2) {
    Point3f a[3] = {Lerp(u0, p[0], p[1]), Lerp(u0, p[1], p[2]),
                    Lerp(u0, p[2], p[3])};
    Point3f b[2] = {Lerp(u1, a[0], a[1]), Lerp(u1, a[1], a[2])};
    return Lerp(u2, b[0], b[1]);
}

inline void SubdivideBezier(const Point3f cp[4], Point3f cpSplit[7]) {
    cpSplit[0] = cp[0];
    cpSplit[1] = (cp[0] + cp[1]) / 2;
    cpSplit[2] = (cp[0] + 2 * cp[1] + cp[2]) / 4;
    cpSplit[3] = (cp[0] + 3 * cp[1] + 3 * cp[2] + cp[3]) / 8;
    cpSplit[4] = (cp[1] + 2 * cp[2] + cp[3]) / 4;
    cpSplit[5] = (cp[2] + cp[3]) / 2;
    cpSplit[6] = cp[3];
}

static Point3f EvalBezier(const Point3f cp[4], Float u,
                          Vector3f *deriv = nullptr) {
    Point3f cp1[3] = {Lerp(u, cp[0], cp[1]), Lerp(u, cp[1], cp[2]),
                      Lerp(u, cp[2], cp[3])};
    Point3f cp2[2] = {Lerp(u, cp1[0], cp1[1]), Lerp(u, cp1[1], cp1[2])};
    if (deriv) *deriv = (Float)3 * (cp2[1] - cp2[0]);
    return Lerp(u, cp2[0], cp2[1]);
}

// Fur Method Definitions
FurCommon::FurCommon(const Point3f c[4], Float width0, Float width1,
                         FurType type, const Normal3f *norm)
    : type(type) {
    width[0] = width0;
    width[1] = width1;
    for (int i = 0; i < 4; ++i)
        cpObj[i] = c[i];
    if (norm) {
        n[0] = Normalize(norm[0]);
        n[1] = Normalize(norm[1]);
        normalAngle = std::acos(Clamp(Dot(n[0], n[1]), 0, 1));
        invSinNormalAngle = 1 / std::sin(normalAngle);
    }
    ++nCurves;
}

std::vector<std::shared_ptr<Shape>> CreateFur(
    const Transform *o2w, const Transform *w2o, bool reverseOrientation,
    const Point3f c[4], Float w0, Float w1, FurType type,
    const Normal3f *norm, int splitDepth) {
    std::vector<std::shared_ptr<Shape>> segments;
    std::shared_ptr<FurCommon> common =
        std::make_shared<FurCommon>(c, w0, w1, type, norm);
    const int nSegments = 1 << splitDepth;
    segments.reserve(nSegments);
    for (int i = 0; i < nSegments; ++i) {
        Float uMin = i / (Float)nSegments;
        Float uMax = (i + 1) / (Float)nSegments;
        segments.push_back(std::make_shared<Fur>(o2w, w2o, reverseOrientation,
                                                   common, uMin, uMax));
        ++nSplitCurves;
    }
    curveBytes += sizeof(FurCommon) + nSegments * sizeof(Fur);
    return segments;
}

Bounds3f Fur::ObjectBound() const {
    // Compute object-space control points for Fur segment, _cpObj_
    Point3f cpObj[4];
    cpObj[0] = BlossomBezier(common->cpObj, uMin, uMin, uMin);
    cpObj[1] = BlossomBezier(common->cpObj, uMin, uMin, uMax);
    cpObj[2] = BlossomBezier(common->cpObj, uMin, uMax, uMax);
    cpObj[3] = BlossomBezier(common->cpObj, uMax, uMax, uMax);
    Bounds3f b =
        Union(Bounds3f(cpObj[0], cpObj[1]), Bounds3f(cpObj[2], cpObj[3]));
    Float width[2] = {Lerp(uMin, common->width[0], common->width[1]),
                      Lerp(uMax, common->width[0], common->width[1])};
    return Expand(b, std::max(width[0], width[1]) * 0.5f);
}

bool Fur::Intersect(const Ray &r, Float *tHit, SurfaceInteraction *isect,
                      bool testAlphaTexture) const {
    ProfilePhase p(isect ? Prof::CurveIntersect : Prof::CurveIntersectP);
    ++nTests;
    // Transform _Ray_ to object space
    Vector3f oErr, dErr;
    Ray ray = (*WorldToObject)(r, &oErr, &dErr);

    // Compute object-space control points for Fur segment, _cpObj_
    Point3f cpObj[4];
    cpObj[0] = BlossomBezier(common->cpObj, uMin, uMin, uMin);
    cpObj[1] = BlossomBezier(common->cpObj, uMin, uMin, uMax);
    cpObj[2] = BlossomBezier(common->cpObj, uMin, uMax, uMax);
    cpObj[3] = BlossomBezier(common->cpObj, uMax, uMax, uMax);

    // Project Fur control points to plane perpendicular to ray

    // Be careful to set the "up" direction passed to LookAt() to equal the
    // vector from the first to the last control points.  In turn, this
    // helps orient the Fur to be roughly parallel to the x axis in the
    // ray coordinate system.
    //
    // In turn (especially for Furs that are approaching stright lines),
    // we get Fur bounds with minimal extent in y, which in turn lets us
    // early out more quickly in recursiveIntersect().
    Vector3f dx = Cross(ray.d, cpObj[3] - cpObj[0]);
    if (dx.LengthSquared() == 0) {
        // If the ray and the vector between the first and last control
        // points are parallel, dx will be zero.  Generate an arbitrary xy
        // orientation for the ray coordinate system so that intersection
        // tests can proceeed in this unusual case.
        Vector3f dy;
        CoordinateSystem(ray.d, &dx, &dy);
    }

    Transform objectToRay = LookAt(ray.o, ray.o + ray.d, dx);
    Point3f cp[4] = {objectToRay(cpObj[0]), objectToRay(cpObj[1]),
                     objectToRay(cpObj[2]), objectToRay(cpObj[3])};

    // Before going any further, see if the ray's bounding box intersects
    // the Fur's bounding box. We start with the y dimension, since the y
    // extent is generally the smallest (and is often tiny) due to our
    // careful orientation of the ray coordinate ysstem above.
    Float maxWidth = std::max(Lerp(uMin, common->width[0], common->width[1]),
                              Lerp(uMax, common->width[0], common->width[1]));
    if (std::max(std::max(cp[0].y, cp[1].y), std::max(cp[2].y, cp[3].y)) +
            0.5f * maxWidth < 0 ||
        std::min(std::min(cp[0].y, cp[1].y), std::min(cp[2].y, cp[3].y)) -
            0.5f * maxWidth > 0)
        return false;

    // Check for non-overlap in x.
    if (std::max(std::max(cp[0].x, cp[1].x), std::max(cp[2].x, cp[3].x)) +
            0.5f * maxWidth < 0 ||
        std::min(std::min(cp[0].x, cp[1].x), std::min(cp[2].x, cp[3].x)) -
            0.5f * maxWidth > 0)
        return false;

    // Check for non-overlap in z.
    Float rayLength = ray.d.Length();
    Float zMax = rayLength * ray.tMax;
    if (std::max(std::max(cp[0].z, cp[1].z), std::max(cp[2].z, cp[3].z)) +
            0.5f * maxWidth < 0 ||
        std::min(std::min(cp[0].z, cp[1].z), std::min(cp[2].z, cp[3].z)) -
            0.5f * maxWidth > zMax)
        return false;

    // Compute refinement depth for Fur, _maxDepth_
    Float L0 = 0;
    for (int i = 0; i < 2; ++i)
        L0 = std::max(
            L0, std::max(
                    std::max(std::abs(cp[i].x - 2 * cp[i + 1].x + cp[i + 2].x),
                             std::abs(cp[i].y - 2 * cp[i + 1].y + cp[i + 2].y)),
                    std::abs(cp[i].z - 2 * cp[i + 1].z + cp[i + 2].z)));

    Float eps =
        std::max(common->width[0], common->width[1]) * .05f;  // width / 20
    auto Log2 = [](Float v) -> int {
        if (v < 1) return 0;
        uint32_t bits = FloatToBits(v);
        // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
        // (With an additional add so get round-to-nearest rather than
        // round down.)
        return (bits >> 23) - 127 + (bits & (1 << 22) ? 1 : 0);
    };
    // Compute log base 4 by dividing log2 in half.
    int r0 = Log2(1.41421356237f * 6.f * L0 / (8.f * eps)) / 2;
    int maxDepth = Clamp(r0, 0, 10);
    ReportValue(refinementLevel, maxDepth);

    return recursiveIntersect(ray, tHit, isect, cp, Inverse(objectToRay), uMin,
                              uMax, maxDepth);
}

bool Fur::recursiveIntersect(const Ray &ray, Float *tHit,
                               SurfaceInteraction *isect, const Point3f cp[4],
                               const Transform &rayToObject, Float u0, Float u1,
                               int depth) const {
    Float rayLength = ray.d.Length();

    if (depth > 0) {
        // Split Fur segment into sub-segments and test for intersection
        Point3f cpSplit[7];
        SubdivideBezier(cp, cpSplit);

        // For each of the two segments, see if the ray's bounding box
        // overlaps the segment before recursively checking for
        // intersection with it.
        bool hit = false;
        Float u[3] = {u0, (u0 + u1) / 2.f, u1};
        // Pointer to the 4 control poitns for the current segment.
        const Point3f *cps = cpSplit;
        for (int seg = 0; seg < 2; ++seg, cps += 3) {
            Float maxWidth =
                std::max(Lerp(u[seg], common->width[0], common->width[1]),
                         Lerp(u[seg + 1], common->width[0], common->width[1]));

            // As above, check y first, since it most commonly lets us exit
            // out early.
            if (std::max(std::max(cps[0].y, cps[1].y),
                         std::max(cps[2].y, cps[3].y)) +
                        0.5 * maxWidth < 0 ||
                std::min(std::min(cps[0].y, cps[1].y),
                         std::min(cps[2].y, cps[3].y)) -
                        0.5 * maxWidth > 0)
                continue;

            if (std::max(std::max(cps[0].x, cps[1].x),
                         std::max(cps[2].x, cps[3].x)) +
                        0.5 * maxWidth < 0 ||
                std::min(std::min(cps[0].x, cps[1].x),
                         std::min(cps[2].x, cps[3].x)) -
                        0.5 * maxWidth > 0)
                continue;

            Float zMax = rayLength * ray.tMax;
            if (std::max(std::max(cps[0].z, cps[1].z),
                         std::max(cps[2].z, cps[3].z)) +
                        0.5 * maxWidth < 0 ||
                std::min(std::min(cps[0].z, cps[1].z),
                         std::min(cps[2].z, cps[3].z)) -
                        0.5 * maxWidth > zMax)
                continue;

            hit |= recursiveIntersect(ray, tHit, isect, cps, rayToObject,
                                      u[seg], u[seg + 1], depth - 1);
            // If we found an intersection and this is a shadow ray,
            // we can exit out immediately.
            if (hit && !tHit) return true;
        }
        return hit;
    } else {
        // Intersect ray with Fur segment

        // Test ray against segment endpoint boundaries

        // Test sample point against tangent perpendicular at Fur start
        Float edge =
            (cp[1].y - cp[0].y) * -cp[0].y + cp[0].x * (cp[0].x - cp[1].x);
        if (edge < 0) return false;

        // Test sample point against tangent perpendicular at Fur end
        edge = (cp[2].y - cp[3].y) * -cp[3].y + cp[3].x * (cp[3].x - cp[2].x);
        if (edge < 0) return false;

        // Compute line $w$ that gives minimum distance to sample point
        Vector2f segmentDirection = Point2f(cp[3]) - Point2f(cp[0]);
        Float denom = segmentDirection.LengthSquared();
        if (denom == 0) return false;
        Float w = Dot(-Vector2f(cp[0]), segmentDirection) / denom;

        // Compute $u$ coordinate of Fur intersection point and _hitWidth_
        Float u = Clamp(Lerp(w, u0, u1), u0, u1);
        Float hitWidth = Lerp(u, common->width[0], common->width[1]);
        Normal3f nHit;
        if (common->type == FurType::Ribbon) {
            // Scale _hitWidth_ based on ribbon orientation
            Float sin0 = std::sin((1 - u) * common->normalAngle) *
                         common->invSinNormalAngle;
            Float sin1 =
                std::sin(u * common->normalAngle) * common->invSinNormalAngle;
            nHit = sin0 * common->n[0] + sin1 * common->n[1];
            hitWidth *= AbsDot(nHit, ray.d) / rayLength;
        }

        // Test intersection point against Fur width
        Vector3f dpcdw;
        Point3f pc = EvalBezier(cp, Clamp(w, 0, 1), &dpcdw);
        Float ptFurDist2 = pc.x * pc.x + pc.y * pc.y;
        if (ptFurDist2 > hitWidth * hitWidth * .25) return false;
        Float zMax = rayLength * ray.tMax;
        if (pc.z < 0 || pc.z > zMax) return false;

        // Compute $v$ coordinate of Fur intersection point
        Float ptFurDist = std::sqrt(ptFurDist2);
        Float edgeFunc = dpcdw.x * -pc.y + pc.x * dpcdw.y;
        Float v = (edgeFunc > 0) ? 0.5f + ptFurDist / hitWidth
                                 : 0.5f - ptFurDist / hitWidth;

        // Compute hit _t_ and partial derivatives for Fur intersection
        if (tHit != nullptr) {
            // FIXME: this tHit isn't quite right for ribbons...
            *tHit = pc.z / rayLength;
            // Compute error bounds for Fur intersection
            Vector3f pError(2 * hitWidth, 2 * hitWidth, 2 * hitWidth);

            // Compute $\dpdu$ and $\dpdv$ for Fur intersection
            Vector3f dpdu, dpdv;
            EvalBezier(common->cpObj, u, &dpdu);
            if (common->type == FurType::Ribbon)
                dpdv = Normalize(Cross(nHit, dpdu)) * hitWidth;
            else {
                // Compute Fur $\dpdv$ for flat and cylinder Furs
                Vector3f dpduPlane = (Inverse(rayToObject))(dpdu);
                Vector3f dpdvPlane =
                    Normalize(Vector3f(-dpduPlane.y, dpduPlane.x, 0)) *
                    hitWidth;
                if (common->type == FurType::Cylinder) {
                    // Rotate _dpdvPlane_ to give cylindrical appearance
                    Float theta = Lerp(v, -90., 90.);
                    Transform rot = Rotate(-theta, dpduPlane);
                    dpdvPlane = rot(dpdvPlane);
                }
                dpdv = rayToObject(dpdvPlane);
            }
            *isect = (*ObjectToWorld)(SurfaceInteraction(
                ray(pc.z), pError, Point2f(u, v), -ray.d, dpdu, dpdv,
                Normal3f(0, 0, 0), Normal3f(0, 0, 0), ray.time, this));
        }
        ++nHits;
        return true;
    }
}

Float Fur::Area() const {
    // Compute object-space control points for Fur segment, _cpObj_
    Point3f cpObj[4];
    cpObj[0] = BlossomBezier(common->cpObj, uMin, uMin, uMin);
    cpObj[1] = BlossomBezier(common->cpObj, uMin, uMin, uMax);
    cpObj[2] = BlossomBezier(common->cpObj, uMin, uMax, uMax);
    cpObj[3] = BlossomBezier(common->cpObj, uMax, uMax, uMax);
    Float width0 = Lerp(uMin, common->width[0], common->width[1]);
    Float width1 = Lerp(uMax, common->width[0], common->width[1]);
    Float avgWidth = (width0 + width1) * 0.5f;
    Float approxLength = 0.f;
    for (int i = 0; i < 3; ++i)
        approxLength += Distance(cpObj[i], cpObj[i + 1]);
    return approxLength * avgWidth;
}

Interaction Fur::Sample(const Point2f &u, Float *pdf) const {
    LOG(FATAL) << "Fur::Sample not implemented.";
    return Interaction();
}

std::vector<std::shared_ptr<Shape>> CreateFurShape(const Transform *o2w,
                                                     const Transform *w2o,
                                                     bool reverseOrientation,
                                                     const ParamSet &params) {
    Float width = params.FindOneFloat("width", 1.f);
    Float width0 = params.FindOneFloat("width0", width);
    Float width1 = params.FindOneFloat("width1", width);

    int ncp;
    const Point3f *cp = params.FindPoint3f("P", &ncp);
    if (ncp != 4) {
        Error(
            "Must provide 4 control points for \"Fur\" primitive. "
            "(Provided %d).",
            ncp);
        return std::vector<std::shared_ptr<Shape>>();
    }

    FurType type;
    std::string furType = params.FindOneString("type", "flat");
    if (furType == "flat")
        type = FurType::Flat;
    else if (furType == "ribbon")
        type = FurType::Ribbon;
    else if (furType == "cylinder")
        type = FurType::Cylinder;
    else {
        Error("Unknown fur type \"%s\".  Using \"flat\".", furType.c_str());
        type = FurType::Cylinder;
    }
    int nnorm;
    const Normal3f *n = params.FindNormal3f("N", &nnorm);
    if (n != nullptr) {
        if (type != FurType::Ribbon) {
            Warning("Fur normals are only used with \"ribbon\" type Furs.");
            n = nullptr;
        } else if (nnorm != 2) {
            Error(
                "Must provide two normals with \"N\" parameter for ribbon "
                "Furs. "
                "(Provided %d).",
                nnorm);
            return std::vector<std::shared_ptr<Shape>>();
        }
    }

    int sd = params.FindOneFloat("splitdepth", 3);

    if (type == FurType::Ribbon && !n) {
        Error(
            "Must provide normals \"N\" at Fur endpoints with ribbon "
            "Furs.");
        return std::vector<std::shared_ptr<Shape>>();
    } else
        return CreateFur(o2w, w2o, reverseOrientation, cp, width0, width1,
                           type, n, sd);
}

}  // namespace pbrt
