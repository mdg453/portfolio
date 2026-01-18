using UnityEngine;

public static class SHCoefficientGenerator
{
    private static Color[][] facePixels;
    private static int faceWidth;
    
    public const int SH_COEFFICIENT_COUNT = 16;

    public static Vector3[] GenerateCoefficients(Cubemap cubemap, int sampleCount)
    {
        PrefetchCubemapFaces(cubemap);

        var shCoefficients = new Vector3[SH_COEFFICIENT_COUNT];
        
        Random.InitState(666); // Determinism for testing.
        
        // Monte Carlo integration.
        for (var i = 0; i < sampleCount; i++)
        {
            var sampleDir = Random.onUnitSphere;

            var sampleColor = SampleCubemap(sampleDir);

            // Make sure we're in linear space.
            if (cubemap.isDataSRGB)
            {
                sampleColor = sampleColor.linear;    
            }

            // Clamp the HDR values to prevent the sun from dominating the entire calculation.
            const float kMaxVal = 10.0f;
            sampleColor.r = Mathf.Min(sampleColor.r, kMaxVal);
            sampleColor.g = Mathf.Min(sampleColor.g, kMaxVal);
            sampleColor.b = Mathf.Min(sampleColor.b, kMaxVal);
            
            // Compute S.H. coefficients, store result in shCoefficients.
            var sh = new float[SH_COEFFICIENT_COUNT];
            EvaluateSHBasis(sampleDir, sh);

            for (var k = 0; k < SH_COEFFICIENT_COUNT; k++)
            {
                shCoefficients[k] += new Vector3(sampleColor.r, sampleColor.g, sampleColor.b) * sh[k];
            }
        }

        // Finalize S.H. coefficients.
        // The probability density of a sample on the unit sphere is 1 / (4 * PI).
        // Monte Carlo Estimator: (1 / N) * Sum(f(x_i) / p(x_i))
        // = (1 / N) * Sum(sampleColor * basis * 4 * PI)
        // = (4 * PI / N) * Sum(sampleColor * basis)
        float factor = (4.0f * Mathf.PI) / sampleCount;
        for (var i = 0; i < SH_COEFFICIENT_COUNT; i++)
        {
            shCoefficients[i] *= factor;
        }
        return shCoefficients;
    }

    private static void PrefetchCubemapFaces(Cubemap cubemap)
    {
        faceWidth = cubemap.width;
        facePixels = new Color[6][];
        
        for (var i = 0; i < 6; i++)
        {
            facePixels[i] = cubemap.GetPixels((CubemapFace)i);
        }
    }

    private static Color SampleCubemap(Vector3 dir)
    {
        DirectionToCubemapFace(dir, out var faceIndex, out var uv);
        
        // Convert UV from [0,1] to pixel coordinates.
        var px = uv.x * faceWidth;
        var py = uv.y * faceWidth;

        // Get integer pixel coordinates and fractional parts for interpolation.
        var x0 = Mathf.FloorToInt(px);
        var y0 = Mathf.FloorToInt(py);
        var tx = px - x0;
        var ty = py - y0;

        // Clamp coordinates to be within bounds.
        x0 = Mathf.Clamp(x0, 0, faceWidth - 1);
        y0 = Mathf.Clamp(y0, 0, faceWidth - 1);
        var x1 = Mathf.Min(x0 + 1, faceWidth - 1);
        var y1 = Mathf.Min(y0 + 1, faceWidth - 1);

        // Get the four neighboring pixel colors.
        var c00 = facePixels[faceIndex][y0 * faceWidth + x0];
        var c10 = facePixels[faceIndex][y0 * faceWidth + x1];
        var c01 = facePixels[faceIndex][y1 * faceWidth + x0];
        var c11 = facePixels[faceIndex][y1 * faceWidth + x1];

        // Bilinear interpolation.
        var a = Color.Lerp(c00, c10, tx);
        var b = Color.Lerp(c01, c11, tx);
        return Color.Lerp(a, b, ty);
    }
    
    private static void DirectionToCubemapFace(Vector3 dir, out int faceIndex, out Vector2 uv)
    {
        var absX = Mathf.Abs(dir.x);
        var absY = Mathf.Abs(dir.y);
        var absZ = Mathf.Abs(dir.z);
        
        if (absX >= absY && absX >= absZ)
        {
            faceIndex = dir.x > 0 ? 0 : 1; // +X or -X
            uv.x = dir.x > 0 ? -dir.z / absX : dir.z / absX;
            uv.y = -dir.y / absX;
        }
        else if (absY >= absX && absY >= absZ)
        {
            faceIndex = dir.y > 0 ? 2 : 3; // +Y or -Y
            uv.x = dir.x / absY;
            uv.y = dir.y > 0 ? dir.z / absY : -dir.z / absY;
        }
        else
        {
            faceIndex = dir.z > 0 ? 4 : 5; // +Z or -Z
            uv.x = dir.z > 0 ? dir.x / absZ : -dir.x / absZ;
            uv.y = -dir.y / absZ;
        }
        
        // Transform uv from [-1,1] to [0,1] range
        uv = (uv + Vector2.one) * 0.5f;
    }

    private static void EvaluateSHBasis(Vector3 dir, float[] sh)
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
}