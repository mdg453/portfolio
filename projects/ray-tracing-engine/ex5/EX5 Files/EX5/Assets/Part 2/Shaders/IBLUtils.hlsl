#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/ImageBasedLighting.hlsl"

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max(float3(1,1,1) * 1.0 - roughness, F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Based on "Physically Based Shading in Call of Duty: Black Ops 3" by Jorge Jimenez
// roughness: Perceptual roughness, a value from 0 (smooth) to 1 (rough).
// NoV: The dot product of the surface normal and the view vector.
// F0: The base reflectivity of the material at normal incidence.
float3 EnvironmentBRDF(float roughness, float NoV, float3 F0)
{
    // Pre-calculated constants derived from the fitting process
    float4 t = float4(1.041667f, 0.475f, 0.018229f, 0.25f);

    // Apply roughness to the first three constants
    t.xyz *= roughness;
    
    // Add the remaining constants
    t += float4(0.0f, 0.0f, -0.015625f, 0.75f);

    // The core of the approximation, which models the BRDF for dielectrics (F0=0.04)
    // t.x corresponds to the scale, t.y to the power, and t.z to the bias.
    float a0 = t.x * min(t.y, exp2(-9.28f * NoV)) + t.z;
    
    // The BRDF for a perfect mirror (F0=1.0) is simply the fourth constant.
    float a1 = t.w;

    // The final result is a linear interpolation between the dielectric BRDF (a0)
    // and the perfect mirror BRDF (a1), using the material's F0 as the interpolant.
    // This is equivalent to: lerp(a0, a1, F0)
    return saturate(a0 + F0 * (a1 - a0));
}

// Taken from https://github.com/SaschaWillems/Vulkan-glTF-PBR/blob/master/data/shaders/genbrdflut.frag
// Based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float2 hammersley(uint i, uint N) 
{
    // Radical inverse based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
    uint bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10;
    return float2(float(i) /float(N), rdi);
}

float3 importanceSampleGGX(float2 Xi, float linearRoughness, float3 N)
{
    float a = linearRoughness * linearRoughness;
    float Phi = 2.0 * PI * Xi.x;
    float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float SinTheta = sqrt(1.0 - CosTheta * CosTheta);
    
    float3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;
    
    float3 UpVector = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 TangentX = normalize(cross(UpVector, N));
    float3 TangentY = cross(N, TangentX);
    return TangentX * H.x + TangentY * H.y + N * H.z;
}

float3 prefilterEnvMap(float roughness, float3 R, TextureCube _EnvTex, SamplerState sampler_EnvTex)
{
    float3 N = R;
    float3 V = R;
    float3 prefilteredColor = float3(0.0, 0.0, 0.0);
    const uint numSamples = 1024; // Reduce to 128 or even 64 if FPS is too low for you. Quality will drop though.
    float totalWeight = 0.0;

    uint width, height, numLODs;
    _EnvTex.GetDimensions(0, width, height, numLODs);
    float imgSize = (float)width;
    
    for (uint i = 0u; i < numSamples; i++) {
        float2 Xi = hammersley(i, numSamples);
        float3 H = importanceSampleGGX(Xi, roughness, N);
        float3 L = 2.0 * dot(V, H) * H - V;
        float NoL = saturate(dot(N, L));
        float NoH = saturate(dot(N, H));
        
        if (NoL > 0.0) {
            float pdf = D_GGX(NoH, roughness) / 4.0 + 0.001;
            float omegaS = 1.0 / (float(numSamples) * pdf);
            float omegaP = 4.0 * PI / (6.0 * imgSize * imgSize);
            float mipLevel = max(0.5 * log2(omegaS / omegaP), 0.0);
            
            prefilteredColor += _EnvTex.SampleLevel(sampler_EnvTex, L, mipLevel).rgb * NoL;
            totalWeight += NoL;
        }
    }
    return prefilteredColor / max(totalWeight, 0.001);
}