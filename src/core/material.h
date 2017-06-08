/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_MATERIAL_H
#define PBRT_CORE_MATERIAL_H

 // Number of bins
#define NUM_SCATTERING_INNER 64
#define NUM_H 64
#define NUM_THETA 64
#define NUM_G 16
#define NUM_BINS 720

// core/material.h*

#include "pbrt.h"
#include "memory.h"

namespace pbrt {
// Storage for precomputed data
// TODO: move these into the appropriate class

extern float scattered[NUM_SCATTERING_INNER][NUM_H][NUM_G][NUM_BINS];
extern float scatteredDist[NUM_SCATTERING_INNER][NUM_H][NUM_G][NUM_BINS];
extern float scatteredM[NUM_SCATTERING_INNER][NUM_THETA][NUM_G][NUM_BINS];
extern float integratedM[NUM_SCATTERING_INNER][NUM_THETA][NUM_G][NUM_BINS];

// TransportMode Declarations
enum class TransportMode { Radiance, Importance };

// Material Declarations
class Material {
  public:
    // Material Interface
    virtual void ComputeScatteringFunctions(SurfaceInteraction *si,
                                            MemoryArena &arena,
                                            TransportMode mode,
                                            bool allowMultipleLobes) const = 0;
    virtual ~Material();
    static void Bump(const std::shared_ptr<Texture<Float>> &d,
                     SurfaceInteraction *si);
	static void initialize(const char *medullaFilename) {
		// Read precomputed medulla profiles (azimuthal)
		FILE *fMedullaN = fopen((std::string(medullaFilename) + "_azimuthal.bin").c_str(), "rb");
		CHECK(fMedullaN != NULL);
		fread(scattered, sizeof(float), NUM_SCATTERING_INNER * NUM_H * NUM_G * NUM_BINS, fMedullaN);
		fread(scatteredDist, sizeof(float), NUM_SCATTERING_INNER * NUM_H * NUM_G * NUM_BINS, fMedullaN);
		fclose(fMedullaN);

		// Read precomputed medulla profiles (longitudinal)
		FILE *fMedullaM = fopen((std::string(medullaFilename) + "_longitudinal.bin").c_str(), "rb");
		CHECK(fMedullaM != NULL);
		fread(scatteredM, sizeof(float), NUM_SCATTERING_INNER * NUM_THETA * NUM_G * NUM_BINS, fMedullaM);
		fread(integratedM, sizeof(float), NUM_SCATTERING_INNER * NUM_THETA * NUM_G * NUM_BINS, fMedullaM);
		fclose(fMedullaM);
	}
};

}  // namespace pbrt

#endif  // PBRT_CORE_MATERIAL_H
