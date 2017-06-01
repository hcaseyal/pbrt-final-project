#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SHAPES_FUR_H
#define PBRT_SHAPES_FUR_H

// shapes/fur.h*
#include "shape.h"

namespace pbrt {
struct FurCommon;

// CurveType Declarations
enum class FurType { Flat, Cylinder, Ribbon };

// FurCommon Declarations
struct FurCommon {
    FurCommon(const Point3f c[4], Float w0, Float w1, FurType type,
                const Normal3f *norm);
    const FurType type;
    Point3f cpObj[4];
    Float width[2];
    Normal3f n[2];
    Float normalAngle, invSinNormalAngle;
};

// Fur Declarations
class Fur : public Shape {
  public:
    // Fur Public Methods
    Fur(const Transform *ObjectToWorld, const Transform *WorldToObject,
          bool reverseOrientation, const std::shared_ptr<FurCommon> &common,
          Float uMin, Float uMax)
        : Shape(ObjectToWorld, WorldToObject, reverseOrientation),
          common(common),
          uMin(uMin),
          uMax(uMax) {}
    Bounds3f ObjectBound() const;
    bool Intersect(const Ray &ray, Float *tHit, SurfaceInteraction *isect,
                   bool testAlphaTexture) const;
    Float Area() const;
    Interaction Sample(const Point2f &u, Float *pdf) const;

  private:
    // Fur Private Methods
    bool recursiveIntersect(const Ray &r, Float *tHit,
                            SurfaceInteraction *isect, const Point3f cp[4],
                            const Transform &rayToObject, Float u0, Float u1,
                            int depth) const;

    // Fur Private Data
    const std::shared_ptr<FurCommon> common;
    const Float uMin, uMax;
};

std::vector<std::shared_ptr<Shape>> CreateFurShape(const Transform *o2w,
                                                     const Transform *w2o,
                                                     bool reverseOrientation,
                                                     const ParamSet &params);

}  // namespace pbrt

#endif  // PBRT_SHAPES_FUR_H
