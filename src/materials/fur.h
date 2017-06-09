#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_FURMATERIAL_H
#define PBRT_MATERIALS_FURMATERIAL_H

// materials/furmaterial.h*
#include "materials/hair.h"
#include "material.h"
#include "pbrt.h"
#include "reflection.h"
#include <array>

namespace pbrt {

// FURMaterial Declarations
class FurMaterial : public Material {
  public:
    // FurMaterial Public Methods
    FurMaterial(const std::shared_ptr<Texture<Spectrum>> &sigma_a,
                 const std::shared_ptr<Texture<Spectrum>> &color,
                 const std::shared_ptr<Texture<Float>> &eumelanin,
                 const std::shared_ptr<Texture<Float>> &pheomelanin,
                 const std::shared_ptr<Texture<Float>> &eta,
                 const std::shared_ptr<Texture<Float>> &beta_m,
                 const std::shared_ptr<Texture<Float>> &beta_n,
                 const std::shared_ptr<Texture<Float>> &alpha,
				 const std::shared_ptr<Texture<Spectrum>> &sigma_c_a,
			     const std::shared_ptr<Texture<Float>> &sigma_m_a,
				 const std::shared_ptr<Texture<Float>> &sigma_m_s,
				 const std::shared_ptr<Texture<Float>> &k,
				 const std::shared_ptr<Texture<Float>> &cuticle_layers,
				 const std::shared_ptr<Texture<Float>> &g)
        : sigma_a(sigma_a),
          color(color),
          eumelanin(eumelanin),
          pheomelanin(pheomelanin),
          eta(eta),
          beta_m(beta_m),
          beta_n(beta_n),
          alpha(alpha), 
		  sigma_c_a(sigma_c_a),
		  sigma_m_a(sigma_m_a),
		  sigma_m_s(sigma_m_s),
		  k(k),
		  cuticle_layers(cuticle_layers),
		  g(g){}
    void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
                                    TransportMode mode,
                                    bool allowMultipleLobes) const;

  private:
    // FurMaterial Private Data
    std::shared_ptr<Texture<Spectrum>> sigma_a, sigma_c_a, color;
    std::shared_ptr<Texture<Float>> eumelanin, pheomelanin, eta;
    std::shared_ptr<Texture<Float>> beta_m, beta_n, alpha;
	std::shared_ptr<Texture<Float>> sigma_m_a, sigma_m_s, k, cuticle_layers, g;
};

FurMaterial *CreateFurMaterial(const TextureParams &mp);

// FurBSDF Constants
static const int pMaxFur = 5;
static const Float SqrtPiOver8Fur = 0.626657069f;

// FurBSDF Declarations
class FurBSDF : public BxDF {
  public:
    // FurBSDF Public Methods
    FurBSDF(Float h, Float eta, const Spectrum &sigma_a, Spectrum sigma_c_a, Float sigma_m_a, Float sigma_m_s, 
		Float beta_m, Float beta_n, Float alpha, Float k, Float cuticle_layers, Float g);
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    Spectrum Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u,
                      Float *pdf, BxDFType *sampledType) const;
    Float Pdf(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;
    static Spectrum SigmaAFromConcentration(Float ce, Float cp);
    static Spectrum SigmaAFromReflectance(const Spectrum &c, Float beta_n);

  private:
    // FurBSDF Private Methods
    std::array<Float, pMaxFur + 1> ComputeApPdf(Float cosThetaO) const;

	Spectrum FurBSDF::computeScatteringLobes(Float thetaI, Float thetaO, Float phiO, std::array<Spectrum, 2> asp, Float gammaT, Float gammaI) const;

    // FurBSDF Private Data
    const Float h, gammaO, eta;
    const Spectrum sigma_a;
    const Float beta_m, beta_n;
    Float stdev_longitudinal[pMaxFur + 1];
	Float stdev_azimuthal[pMaxFur + 1];
    Float s;
    Float sin2kAlpha[3], cos2kAlpha[3];
	Float alphas[pMaxFur + 1];
	const Spectrum sigma_c_a;
	const Float sigma_m_a;
	const Float sigma_m_s;
	const Float k; // medullary index (rel. radius length)
	const Float cuticle_layers;
	const Float g;
};

}  // namespace pbrt

#endif  // PBRT_MATERIALS_FUR_H
