// materials/fur.cpp*
#include <array>
#include <numeric>

#include "interaction.h"
#include "materials/fur.h"
#include "paramset.h"
#include "reflection.h"
#include "sampling.h"
#include "spectrum.h"
#include "texture.h"
#include "textures/constant.h"

namespace pbrt {

// fur Local Declarations
inline Float I0(Float x), LogI0(Float x);
inline Float Theta(int p, Float thetaI, const Float alphas[]);
inline Float TrimmedLogistic(Float x, Float s, Float a, Float b);

// fur Local Functions
static int indexFromValue(Float value, Float rangeSize, Float minRange, int numSteps) {
	return roundf((value - minRange) / (rangeSize / (numSteps - 1)));
}

// Mp(θi, θr) = G(θr; −θi + αp, βp),
static Float Mp(Float thetaI, Float thetaO, const Float alphas[], int p, Float stdev, Float sigma_m_s, Float k, Float g) {
	Float mp;
	if (p == 3 || p == 4) {
		int num_scattering_inner = indexFromValue(sigma_m_s / k, 20, 0, NUM_SCATTERING_INNER);
		int num_theta = indexFromValue(thetaI, Pi, Pi / 2, NUM_THETA);  // Theta is supposed to be between -pi/2 and pi/2?
		int num_g = indexFromValue(g, 8, 0, NUM_G);
		int num_bin = indexFromValue(thetaO, Pi, Pi / 2, NUM_BINS);
		mp = scatteredM[num_scattering_inner][num_theta][num_g][num_bin];
	}
	else {
		Float dtheta = thetaO - Theta(p, thetaI, alphas);
		// Remap _dtheta_ to $[-\pi,\pi]$
		while (dtheta > Pi) dtheta -= 2 * Pi;
		while (dtheta < -Pi) dtheta += 2 * Pi;
		mp = TrimmedLogistic(dtheta, stdev, -Pi, Pi);
	}
	return mp;
}

inline Float I0(Float x) {
    Float val = 0;
    Float x2i = 1;
    int ifact = 1;
    int i4 = 1;
    // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
    for (int i = 0; i < 10; ++i) {
        if (i > 1) ifact *= i;
        val += x2i / (i4 * Sqr(ifact));
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}

inline Float LogI0(Float x) {
    if (x > 12)
        return x + 0.5 * (-std::log(2 * Pi) + std::log(1 / x) + 1 / (8 * x));
    else
        return std::log(I0(x));
}

static std::array<Spectrum, 2> attenuation_scattered(Float cosThetaO, Float eta, Float h, Spectrum T, Float cuticle_layers, Float k, Spectrum T_s) {
	std::array<Spectrum, 2> ap;
	// Compute $p=3$ attenuation at initial cylinder intersection
	Float cosGammaO = SafeSqrt(1 - h * h);
	Float cosTheta = cosThetaO * cosGammaO;
	Float dielectric = FrDielectric(cosTheta, 1.f, eta);
	Float f = (cuticle_layers * dielectric) / (1 + (cuticle_layers - 1) * dielectric);
	ap[0] = f * T_s;

	// Compute $p=4$ attenuation term
	ap[1] = ap[0] * (1 - f) * T;
	return ap;
}

static std::array<Spectrum, pMaxFur + 1> Ap(Float cosThetaO, Float eta, Float h,
                                         const Spectrum &T, Float cuticle_layers, Float k, const Spectrum &T_s) {
	std::array<Spectrum, pMaxFur + 1> ap;
	// Compute $p=0$ attenuation at initial cylinder intersection
	Float cosGammaO = SafeSqrt(1 - h * h);
	Float cosTheta = cosThetaO * cosGammaO;
	Float dielectric = FrDielectric(cosTheta, 1.f, eta);
	Float f = (cuticle_layers * dielectric) / (1 + (cuticle_layers - 1) * dielectric);
	ap[0] = f;

	// Compute $p=1$ attenuation term
	ap[1] = Sqr(1 - f) * T;
	ap[2] = ap[1] * T * f;

	// Now do scattered lobes
	std::array<Spectrum, 2> scatteredAttenuations = attenuation_scattered(cosThetaO, eta, h, T, cuticle_layers, k, T_s);
	ap[3] = scatteredAttenuations[0];
	ap[4] = scatteredAttenuations[1];

	// Compute attenuation term accounting for remaining orders of scattering
	ap[5] = ap[2] * f * T / (Spectrum(1.f) - T * f);
	return ap;
}

inline Float Theta(int p, Float thetaI, const Float alphas[]) {
	return -thetaI + alphas[p];
}

inline Float Phi(int p, Float gammaO, Float gammaT) {
    return 2 * p * gammaT - 2 * gammaO + p * Pi;
}

inline Float Logistic(Float x, Float s) {
    x = std::abs(x);
    return std::exp(-x / s) / (s * Sqr(1 + std::exp(-x / s)));
}

inline Float LogisticCDF(Float x, Float s) {
    return 1 / (1 + std::exp(-x / s));
}

inline Float TrimmedLogistic(Float x, Float s, Float a, Float b) {
    CHECK_LT(a, b);
    return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

inline Float Np(Float phi, int p, Float stdev, Float gammaO, Float gammaT, Float h, Float k, Float sigma_m_s, Float g) {
	int num_scattering_inner = indexFromValue(sigma_m_s / k, 20, 0, NUM_SCATTERING_INNER);
	int num_g = indexFromValue(g, 8, 0, NUM_G);
	Float np;
	if (p == 3 || p == 4) {
		//azimuthal
		Float gammaI = SafeASin(h);
		int num_h = indexFromValue(h / k, 2, -1, NUM_H);
		float chunk = (gammaT - gammaI) + (p - 3) * (Pi + 2 * gammaT);

		int num_bin_s = indexFromValue(phi - chunk, Pi, Pi / 2, NUM_BINS);
		np = scattered[num_scattering_inner][num_h][num_g][num_bin_s];
	} else {
		Float dphi = phi - Phi(p, gammaO, gammaT);
		// Remap _dphi_ to $[-\pi,\pi]$
		while (dphi > Pi) dphi -= 2 * Pi;
		while (dphi < -Pi) dphi += 2 * Pi;
		np = TrimmedLogistic(dphi, stdev, -Pi, Pi);
	}
	return np;
}

static Float SampleTrimmedLogistic(Float u, Float s, Float a, Float b) {
    CHECK_LT(a, b);
    Float k = LogisticCDF(b, s) - LogisticCDF(a, s);
    Float x = -s * std::log(1 / (u * k + LogisticCDF(a, s)) - 1);
    CHECK(!std::isnan(x));
    return Clamp(x, a, b);
}

// furMaterial Method Definitions
void FurMaterial::ComputeScatteringFunctions(SurfaceInteraction *si,
                                              MemoryArena &arena,
                                              TransportMode mode,
                                              bool allowMultipleLobes) const {
    Float bm = Radians(beta_m->Evaluate(*si));
    Float bn = Radians(beta_n->Evaluate(*si));
    Float a = Radians(alpha->Evaluate(*si));
    Float e = eta->Evaluate(*si);

    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si, e);

    Spectrum sig_a;
    if (sigma_a)
        sig_a = sigma_a->Evaluate(*si).Clamp();
    else if (color) {
        Spectrum c = color->Evaluate(*si).Clamp();
        sig_a = FurBSDF::SigmaAFromReflectance(c, bn);
    } else {
        CHECK(eumelanin || pheomelanin);
        sig_a = FurBSDF::SigmaAFromConcentration(
            std::max(Float(0), eumelanin ? eumelanin->Evaluate(*si) : 0),
            std::max(Float(0), pheomelanin ? pheomelanin->Evaluate(*si) : 0));
    }
	Spectrum sigma_c_a_ = sigma_c_a->Evaluate(*si);
	Float rgb[3];
	Float sigma_m_a_ = sigma_m_a->Evaluate(*si);
	Float sigma_m_s_ = sigma_m_s->Evaluate(*si);

    // Offset along width
    Float h = -1 + 2 * si->uv[1];
	Float k_ = k->Evaluate(*si);
	Float cuticle_layers_ = cuticle_layers->Evaluate(*si);
	Float g_ = g->Evaluate(*si);
    si->bsdf->Add(ARENA_ALLOC(arena, FurBSDF)(h, e, sig_a, sigma_c_a_, sigma_m_a_, sigma_m_s_, bm, bn, a, k_, cuticle_layers_, g_));
}

FurMaterial *CreateFurMaterial(const TextureParams &mp) {
    std::shared_ptr<Texture<Spectrum>> sigma_a =
        mp.GetSpectrumTextureOrNull("sigma_a");
	std::shared_ptr<Texture<Spectrum>> sigma_c_a =
		mp.GetSpectrumTextureOrNull("sigma_c_a");
	CHECK(sigma_c_a != NULL);
    std::shared_ptr<Texture<Spectrum>> color =
        mp.GetSpectrumTextureOrNull("color");
    std::shared_ptr<Texture<Float>> eumelanin =
        mp.GetFloatTextureOrNull("eumelanin");
    std::shared_ptr<Texture<Float>> pheomelanin =
        mp.GetFloatTextureOrNull("pheomelanin");
    if (sigma_a) {
        if (color)
            Warning(
                "Ignoring \"color\" parameter since \"sigma_a\" was provided.");
        if (eumelanin)
            Warning(
                "Ignoring \"eumelanin\" parameter since \"sigma_a\" was "
                "provided.");
        if (pheomelanin)
            Warning(
                "Ignoring \"pheomelanin\" parameter since \"sigma_a\" was "
                "provided.");
    } else if (color) {
        if (sigma_a)
            Warning(
                "Ignoring \"sigma_a\" parameter since \"color\" was provided.");
        if (eumelanin)
            Warning(
                "Ignoring \"eumelanin\" parameter since \"color\" was "
                "provided.");
        if (pheomelanin)
            Warning(
                "Ignoring \"pheomelanin\" parameter since \"color\" was "
                "provided.");
    } else if (eumelanin || pheomelanin) {
        if (sigma_a)
            Warning(
                "Ignoring \"sigma_a\" parameter since "
                "\"eumelanin\"/\"pheomelanin\" was provided.");
        if (color)
            Warning(
                "Ignoring \"color\" parameter since "
                "\"eumelanin\"/\"pheomelanin\" was provided.");
    } else {
        // Default: brown-ish Fur.
        sigma_a = std::make_shared<ConstantTexture<Spectrum>>(
            FurBSDF::SigmaAFromConcentration(1.3, 0.));
    }

    std::shared_ptr<Texture<Float>> eta = mp.GetFloatTexture("eta", 1.49f);
    std::shared_ptr<Texture<Float>> beta_m = mp.GetFloatTexture("beta_m", 9.45f);
    std::shared_ptr<Texture<Float>> beta_n = mp.GetFloatTexture("beta_n", 17.63f);
    std::shared_ptr<Texture<Float>> alpha = mp.GetFloatTexture("alpha", 2.64f);
	std::shared_ptr<Texture<Float>> sigma_m_a = mp.GetFloatTexture("sigma_m_a", 0.21f);
	std::shared_ptr<Texture<Float>> sigma_m_s = mp.GetFloatTexture("sigma_m_s", 3.15f);
	std::shared_ptr<Texture<Float>> k = mp.GetFloatTexture("k", 0.86f);
	std::shared_ptr<Texture<Float>> cuticle_layers = mp.GetFloatTexture("cuticle_layers", 0.68f);
	std::shared_ptr<Texture<Float>> g = mp.GetFloatTexture("g", 0.79f);
    return new FurMaterial(sigma_a, color, eumelanin, pheomelanin, eta, beta_m,
                            beta_n, alpha, sigma_c_a, sigma_m_a, sigma_m_s, k, cuticle_layers, g);
}

// FurBSDF Method Definitions
FurBSDF::FurBSDF(Float h, Float eta, const Spectrum &sigma_a, Spectrum sigma_c_a, Float sigma_m_a, Float sigma_m_s,
	Float beta_m, Float beta_n, Float alpha, Float k, Float cuticle_layers, Float g)
    : BxDF(BxDFType(BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION)),
      h(h),
      gammaO(SafeASin(h)),
      eta(eta),
      sigma_a(sigma_a),
	  sigma_c_a(sigma_c_a),
	  sigma_m_a(sigma_m_a),
	  sigma_m_s(sigma_m_s),
      beta_m(beta_m),
      beta_n(beta_n),
	  k(k),
	  cuticle_layers(cuticle_layers),
	  g(g){
    CHECK(h >= -1 && h <= 1);
	stdev_azimuthal[0] = beta_n;
	stdev_azimuthal[1] = sqrt(2) * beta_n;
	stdev_azimuthal[2] = sqrt(3) * beta_n;

    stdev_longitudinal[0] = beta_m;
    stdev_longitudinal[1] = beta_m / 2;
    stdev_longitudinal[2] = 3 * beta_m / 2;
	
	// TRRT
	stdev_azimuthal[pMaxFur] = 2 * beta_n;
	stdev_longitudinal[pMaxFur] = 5 * beta_m / 2;

    // Compute azimuthal logistic scale factor from $\beta_n$
    s = SqrtPiOver8Fur *
        (0.265f * beta_n + 1.194f * Sqr(beta_n) + 5.372f * Pow<22>(beta_n));
    CHECK(!std::isnan(s));

	// alphas for each lobe
	alphas[0] = alpha;
	alphas[1] = -alpha / 2;
	alphas[2] = -3 * alpha / 2;
	alphas[pMaxFur] = -5 * alpha / 2;

    // Compute $\alpha$ terms for Fur scales
    sin2kAlpha[0] = std::sin(Radians(alpha));
    cos2kAlpha[0] = SafeSqrt(1 - Sqr(sin2kAlpha[0]));
	for (int i = 1; i < 3; ++i) {
		sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
		cos2kAlpha[i] = Sqr(cos2kAlpha[i - 1]) - Sqr(sin2kAlpha[i - 1]);
	}
}

Spectrum FurBSDF::f(const Vector3f &wo, const Vector3f &wi) const {
    // Compute Fur coordinate system terms related to _wo_
    Float sinThetaO = wo.x;
    Float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
	Float thetaO = std::asinf(sinThetaO);
    Float phiO = std::atan2(wo.z, wo.y);

    // Compute Fur coordinate system terms related to _wi_
    Float sinThetaI = wi.x;
    Float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
	Float thetaI = std::asinf(sinThetaI);
    Float phiI = std::atan2(wi.z, wi.y);

    // Compute $\cos \thetat$ for refracted ray
    Float sinThetaT = sinThetaO / eta;
    Float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
    Float sinGammaT = h / etap;
    Float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
    Float gammaT = SafeASin(sinGammaT);

    // Compute the transmittance _T_ of a single path through the cylinder
	Float s_m;
	Float term = pow(k, 2) - pow(sinGammaT, 2);
	if (term < 0) {
		s_m = 0;
	}
	else {
		s_m = std::sqrt(term);
	}
	Float s_c = cosGammaT - s_m;

	// TODO: stuff below is incorrect
	Spectrum numerator = -1 * (2 * s_c * sigma_c_a + 2 * s_m * (sigma_m_a + sigma_m_s));
	//printf("s_c: %f, s_m: %f \n", s_c, s_m);
	Float thetaD = (thetaO - thetaI) / 2;
	Float denom = cosf(thetaD);
	Spectrum T;
	if (denom > 0) {
		// TODO: why is transmittance so low?
		T = Exp(numerator / denom);
	}
	else {
		T = Spectrum(0.f);
		printf("Transmittance 0!\n");
	}

	Spectrum T_s;
	Spectrum numerator_s = -1 * ((s_c + 1 + k) * sigma_c_a + k * sigma_m_a);
	if (denom > 0) {
		T_s = Exp(numerator / denom);
	}
	else {
		T_s = Spectrum(0.f);
		printf("Transmittance 0!\n");
	}

    // Evaluate Fur BSDF
    Float phi = phiI - phiO;
    std::array<Spectrum, pMaxFur + 1> ap = Ap(cosThetaO, eta, h, T, cuticle_layers, k, T_s);

	Float gammaI = SafeASin(h);
    Spectrum fsum(0.);
    for (int p = 0; p < pMaxFur; ++p) {
		// Compute reflected angle
		Float mp = Mp(thetaI, thetaO, alphas, p, stdev_longitudinal[p], sigma_m_s, k, g);
		Spectrum apVal = ap[p];
        Float np = Np(phi, p, stdev_azimuthal[p], gammaO, gammaT, h, k, sigma_m_s, g);
		fsum += mp * apVal * np;
		if (std::isnan(fsum.y())) {
			printf("\napval %f p %f np %f mp %f \n\n", apVal.y(), p, np, mp);
		}
		CHECK(!std::isinf(fsum.y()));
		CHECK(!std::isnan(fsum.y()));
    }
	// Compute contribution of remaining terms
	fsum += Mp(thetaI, thetaO, alphas, pMaxFur, stdev_longitudinal[pMaxFur], sigma_m_s, k, g) * ap[pMaxFur] /
		(2.f * Pi);

	//fsum += computeScatteringLobes(thetaI, thetaO, phiO, ap, gammaT, gammaI);
	// TODO: Paper says to divide by Sqr(cosThetaI), is that right?
	if (abs(cosThetaI) > 0) {
		fsum /= Sqr(cosThetaI);
	}

	// TODO: do we need below?
	if (AbsCosTheta(wi) > 0) {
		//fsum /= AbsCosTheta(wi);
	}

	if (std::isnan(fsum.y())) {
		printf("\nmp %f ap %f \n", Mp(thetaI, thetaO, alphas, pMaxFur, stdev_longitudinal[pMaxFur], sigma_m_s, k, g), ap[pMaxFur]);
	}
	CHECK(!std::isinf(fsum.y()));
	CHECK(!std::isnan(fsum.y()));
	
    return fsum;
}

// TODO: we can precompute most of this floating point stuff
Spectrum FurBSDF::computeScatteringLobes(Float thetaI, Float thetaO, Float phiO, std::array<Spectrum, 2> asp, Float gammaT, Float gammaI) const {
	// Longitudinal
	int num_scattering_inner = indexFromValue(sigma_m_s / k, 20, 0, NUM_SCATTERING_INNER);
	int num_g = indexFromValue(g, 8, 0, NUM_G);
	int num_theta = indexFromValue(thetaI, Pi, Pi / 2, NUM_THETA);  // Theta is supposed to be between -pi/2 and pi/2?
	int num_bin = indexFromValue(thetaO, Pi, Pi / 2, NUM_BINS);
	float mp = scatteredM[num_scattering_inner][num_theta][num_g][num_bin];

	//azimuthal
	int num_h = indexFromValue(h / k, 2, -1, NUM_H);
	float chunk = (gammaT - gammaI);

	int num_bin_s = indexFromValue(phiO - chunk, Pi, Pi / 2, NUM_BINS); 
	float dsp = scattered[num_scattering_inner][num_h][num_g][num_bin_s];
	Spectrum np = asp[0] * dsp; 
	Spectrum firstLobe = mp * np;

	float chunk2 = 3 * gammaT - gammaI + Pi;

	int num_bin_s_second = indexFromValue(phiO - chunk2, Pi, Pi / 2, NUM_BINS);
	float dsp2 = scattered[num_scattering_inner][num_h][num_g][num_bin_s_second];
	Spectrum np2 = asp[1] * dsp2;
	Spectrum secondLobe = mp * np2;

	return firstLobe + secondLobe;
}

std::array<Float, pMaxFur + 1> FurBSDF::ComputeApPdf(Float cosThetaO) const {
    // Compute array of $A_p$ values for _cosThetaO_
	Float sinThetaO = SafeSqrt(1 - cosThetaO * cosThetaO);
	Float thetaO = std::asinf(sinThetaO);

    // Compute $\cos \thetat$ for refracted ray
    Float sinThetaT = sinThetaO / eta;
    Float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
    Float sinGammaT = h / etap;
    Float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
    Float gammaT = SafeASin(sinGammaT);

	// Compute the transmittance _T_ of a single path through the cylinder
	Spectrum T = Exp(-sigma_a * (2 * cosGammaT / cosThetaT));

	Float s_m;
	Float term = pow(k, 2) - pow(sinGammaT, 2);
	if (term < 0) {
		s_m = 0;
	}
	else {
		s_m = std::sqrt(term);
	}
	Float s_c = cosGammaT - s_m;

	Spectrum numerator = -1 * (2 * s_c * sigma_c_a + 2 * s_m * (sigma_m_a + sigma_m_s));
	// TODO: How to calculate thetaI?
	Float thetaI = 0.f;
	Float thetaD = (thetaO - thetaI) / 2;
	Float denom = cosf(thetaD);

	Spectrum T_s;
	Spectrum numerator_s = -1 * ((s_c + 1 + k) * sigma_c_a + k * sigma_m_a);
	if (denom > 0) {
		T_s = Exp(numerator / denom);
	} else {
		T_s = Spectrum(0.f);
	}

    std::array<Spectrum, pMaxFur + 1> ap = Ap(cosThetaO, eta, h, T, cuticle_layers, k, T_s);


    // Compute $A_p$ PDF from individual $A_p$ terms
    std::array<Float, pMaxFur + 1> apPdf;
    Float sumY =
        std::accumulate(ap.begin(), ap.end(), Float(0),
                        [](Float s, const Spectrum &ap) { return s + ap.y(); });
    for (int i = 0; i <= pMaxFur; ++i) apPdf[i] = ap[i].y() / sumY;
    return apPdf;
}

Spectrum FurBSDF::Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u2,
                            Float *pdf, BxDFType *sampledType) const {

    // Compute Fur coordinate system terms related to _wo_
    Float sinThetaO = wo.x;
    Float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
	Float thetaO = std::asinf(sinThetaO);
    Float phiO = std::atan2(wo.z, wo.y);

    // Derive four random samples from _u2_
    Point2f u[2] = {DemuxFloat(u2[0]), DemuxFloat(u2[1])};

    // Determine which term $p$ to sample for Fur scattering
    std::array<Float, pMaxFur + 1> apPdf = ComputeApPdf(cosThetaO);
    int p;
    for (p = 0; p < pMaxFur; ++p) {
        if (u[0][0] < apPdf[p]) break;
        u[0][0] -= apPdf[p];
    }

    // Sample $M_p$ to compute $\thetai$
    u[1][0] = std::max(u[1][0], Float(1e-5));
    Float cosTheta =
        1 + stdev_longitudinal[p] * std::log(u[1][0] + (1 - u[1][0]) * std::exp(-2 / stdev_longitudinal[p]));
    Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    Float cosPhi = std::cos(2 * Pi * u[1][1]);
    Float sinThetaI = -cosTheta * sinThetaO + sinTheta * cosPhi * cosThetaO;
	Float thetaI = std::asinf(sinThetaI);
    Float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));

    // Update sampled $\sin \thetai$ and $\cos \thetai$ to account for scales
    Float sinThetaIp = sinThetaI, cosThetaIp = cosThetaI;
    if (p == 0) {
        sinThetaIp = sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1];
        cosThetaIp = cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1];
    } else if (p == 1) {
        sinThetaIp = sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0];
        cosThetaIp = cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0];
    } else if (p == 2) {
        sinThetaIp = sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2];
        cosThetaIp = cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2];
    }
    sinThetaI = sinThetaIp;
    cosThetaI = cosThetaIp;

    // Sample $N_p$ to compute $\Delta\phi$

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
    Float sinGammaT = h / etap;
    Float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
    Float gammaT = SafeASin(sinGammaT);
    Float dphi;
    if (p < pMaxFur)
        dphi =
            Phi(p, gammaO, gammaT) + SampleTrimmedLogistic(u[0][1], s, -Pi, Pi);
    else
        dphi = 2 * Pi * u[0][1];

    // Compute _wi_ from sampled Fur scattering angles
    Float phiI = phiO + dphi;
    *wi = Vector3f(sinThetaI, cosThetaI * std::cos(phiI),
                   cosThetaI * std::sin(phiI));

    // Compute PDF for sampled Fur scattering direction _wi_
    *pdf = 0;
    for (int p = 0; p < pMaxFur; ++p) {
        // Compute $\sin \thetai$ and $\cos \thetai$ terms accounting for scales
        Float sinThetaIp, cosThetaIp;
        if (p == 0) {
            sinThetaIp = sinThetaI * cos2kAlpha[1] + cosThetaI * sin2kAlpha[1];
            cosThetaIp = cosThetaI * cos2kAlpha[1] - sinThetaI * sin2kAlpha[1];
        }

        // Handle remainder of $p$ values for Fur scale tilt
        else if (p == 1) {
            sinThetaIp = sinThetaI * cos2kAlpha[0] - cosThetaI * sin2kAlpha[0];
            cosThetaIp = cosThetaI * cos2kAlpha[0] + sinThetaI * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetaIp = sinThetaI * cos2kAlpha[2] - cosThetaI * sin2kAlpha[2];
            cosThetaIp = cosThetaI * cos2kAlpha[2] + sinThetaI * sin2kAlpha[2];
        } else {
            sinThetaIp = sinThetaI;
            cosThetaIp = cosThetaI;
        }

        // Handle out-of-range $\cos \thetai$ from scale adjustment
        cosThetaIp = std::abs(cosThetaIp);
		*pdf += Mp(thetaI, thetaO, alphas, p, stdev_longitudinal[p], sigma_m_s, k, g) * apPdf[p] *
			Np(dphi, p, s, gammaO, gammaT, h, k, sigma_m_s, g);
    }
    *pdf += Mp(thetaI, thetaO, alphas, p, stdev_longitudinal[p], sigma_m_s, k, g) *
            apPdf[pMaxFur] * (1 / (2 * Pi));
    // if (std::abs(wi->x) < .9999) CHECK_NEAR(*pdf, Pdf(wo, *wi), .01);
    return f(wo, *wi);
}

Float FurBSDF::Pdf(const Vector3f &wo, const Vector3f &wi) const {
    // Compute Fur coordinate system terms related to _wo_
    Float sinThetaO = wo.x;
    Float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
	Float thetaO = std::asinf(sinThetaO);
    Float phiO = std::atan2(wo.z, wo.y);

    // Compute Fur coordinate system terms related to _wi_
    Float sinThetaI = wi.x;
    Float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
	Float thetaI = std::asinf(sinThetaI);
    Float phiI = std::atan2(wi.z, wi.y);

    // Compute $\cos \thetat$ for refracted ray
    Float sinThetaT = sinThetaO / eta;
    Float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
    Float sinGammaT = h / etap;
    Float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
    Float gammaT = SafeASin(sinGammaT);

    // Compute PDF for $A_p$ terms
    std::array<Float, pMaxFur + 1> apPdf = ComputeApPdf(cosThetaO);

    // Compute PDF sum for Fur scattering events
    Float phi = phiI - phiO;
    Float pdf = 0;
    for (int p = 0; p < pMaxFur; ++p) {
        // Compute $\sin \thetai$ and $\cos \thetai$ terms accounting for scales
        Float sinThetaIp, cosThetaIp;
        if (p == 0) {
            sinThetaIp = sinThetaI * cos2kAlpha[1] + cosThetaI * sin2kAlpha[1];
            cosThetaIp = cosThetaI * cos2kAlpha[1] - sinThetaI * sin2kAlpha[1];
        }

        // Handle remainder of $p$ values for Fur scale tilt
        else if (p == 1) {
            sinThetaIp = sinThetaI * cos2kAlpha[0] - cosThetaI * sin2kAlpha[0];
            cosThetaIp = cosThetaI * cos2kAlpha[0] + sinThetaI * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetaIp = sinThetaI * cos2kAlpha[2] - cosThetaI * sin2kAlpha[2];
            cosThetaIp = cosThetaI * cos2kAlpha[2] + sinThetaI * sin2kAlpha[2];
        } else {
            sinThetaIp = sinThetaI;
            cosThetaIp = cosThetaI;
        }

        // Handle out-of-range $\cos \thetai$ from scale adjustment
        cosThetaIp = std::abs(cosThetaIp);
        pdf += Mp(thetaI, thetaO, alphas, p, stdev_longitudinal[p], sigma_m_s, k, g) *
               apPdf[p] * Np(phi, p, s, gammaO, gammaT, h, k, sigma_m_s, g);
    }
    pdf += Mp(thetaI, thetaO, alphas, pMaxFur, stdev_longitudinal[pMaxFur], sigma_m_s, k, g) *
           apPdf[pMaxFur] * (1 / (2 * Pi));
    return pdf;
}

std::string FurBSDF::ToString() const {
    return StringPrintf(
        "[ Fur h: %f gammaO: %f eta: %f beta_m: %f beta_n: %f "
        "v[0]: %f s: %f sigma_a: ", h, gammaO, eta, beta_m, beta_n,
        stdev_longitudinal[0], s) +
        sigma_a.ToString() +
        std::string("  ]");
}

Spectrum FurBSDF::SigmaAFromConcentration(Float ce, Float cp) {
    Float sigma_a[3];
    Float eumelaninSigmaA[3] = {0.419f, 0.697f, 1.37f};
    Float pheomelaninSigmaA[3] = {0.187f, 0.4f, 1.05f};
    for (int i = 0; i < 3; ++i)
        sigma_a[i] = (ce * eumelaninSigmaA[i] + cp * pheomelaninSigmaA[i]);
    return Spectrum::FromRGB(sigma_a);
}

Spectrum FurBSDF::SigmaAFromReflectance(const Spectrum &c, Float beta_n) {
    Spectrum sigma_a;
    for (int i = 0; i < Spectrum::nSamples; ++i)
        sigma_a[i] = Sqr(std::log(c[i]) /
                         (5.969f - 0.215f * beta_n + 2.532f * Sqr(beta_n) -
                          10.73f * Pow<3>(beta_n) + 5.574f * Pow<4>(beta_n) +
                          0.245f * Pow<5>(beta_n)));
    return sigma_a;
}

}  // namespace pbrt
