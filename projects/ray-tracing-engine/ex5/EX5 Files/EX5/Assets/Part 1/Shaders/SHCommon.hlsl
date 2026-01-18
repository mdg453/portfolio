#define NUM_COEFFS 16
#ifndef PI
#define PI 3.1415926535
#endif

// Returns the basis for SH for the given direction. dir must be normalized!
void EvaluateSHBasis(float3 dir, out float sh[NUM_COEFFS])
{
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x*x, y2 = y*y, z2 = z*z;

    // Band 0
    sh[0]  = 0.282095f;
    
    // Band 1
    sh[1]  = 0.488603f * y;
    sh[2]  = 0.488603f * z;
    sh[3]  = 0.488603f * x;
    
    // Band 2
    sh[4]  = 1.092548f * x * y;
    sh[5]  = 1.092548f * y * z;
    sh[6]  = 0.315392f * (3.0f * z2 - 1.0f);
    sh[7]  = 1.092548f * x * z;
    sh[8]  = 0.546274f * (x2 - y2);
    
    // Band 3
    sh[9]  = 0.590044f * y * (3.0f * x2 - y2);
    sh[10] = 2.890611f * x * y * z;
    sh[11] = 0.457046f * y * (5.0f * z2 - 1.0f);
    sh[12] = 0.373176f * z * (5.0f * z2 - 3.0f);
    sh[13] = 0.457046f * x * (5.0f * z2 - 1.0f);
    sh[14] = 1.445306f * z * (x2 - y2);
    sh[15] = 0.590044f * x * (x2 - 3.0f * y2);
}

// Helper function to convert a UV coordinate on an equirectangular map
// into a 3D direction vector.
float3 EquirectangularUVToDirection(float2 uv)
{
    // Azimuthal angle (around the Y axis)
    float phi = (uv.x - 0.5) * 2.0 * PI;
    
    // Polar angle (from the Y axis pole)
    float theta = (0.5 - uv.y) * PI;

    float3 dir;
    dir.y = sin(theta);
    float cosTheta = cos(theta);
    dir.x = cosTheta * sin(phi); // This points to +X when phi is 90 degrees (right)
    dir.z = cosTheta * cos(phi); // This points to +Z when phi is 0 (center)
    
    return normalize(dir);
}

float3 RenderSHLighting(float3 dir, StructuredBuffer<float3> shCoefficients)
{
    float sh[NUM_COEFFS];
    EvaluateSHBasis(dir, sh);

    float3 color = float3(0, 0, 0);
    // Diffuse convolution factors per band:
    // L0: PI, L1: 2PI/3, L2: PI/4, L3: PI/6
    float factors[4] = { PI, (2.0 * PI) / 3.0, PI * 0.25, PI / 6.0 };

    for (int i = 0; i < NUM_COEFFS; i++)
    {
        // Determine band index (0 to 3)
        int band = 0;
        if (i >= 9) band = 3;
        else if (i >= 4) band = 2;
        else if (i >= 1) band = 1;

        color += shCoefficients[i] * sh[i] * factors[band];
    }

    return color;
}

float3 RenderSHLighting(float2 uv, StructuredBuffer<float3> shCoefficients)
{
    float3 dir = EquirectangularUVToDirection(uv);
    return RenderSHLighting(dir, shCoefficients);
}