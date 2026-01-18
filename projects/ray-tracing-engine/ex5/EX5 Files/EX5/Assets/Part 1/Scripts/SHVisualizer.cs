using UnityEngine;

[ExecuteAlways]
public class SHVisualizer : MonoBehaviour
{
    [SerializeField] private ComputeShader shRenderShader;
    [SerializeField] private RenderTexture targetRenderTexture;
    
    private Vector3[] shCoefficients;

    public void SetSHCoefficients(Vector3[] shCoefficients)
    {
        this.shCoefficients = shCoefficients;
    }
    
    private void RenderToTexture()
    {
        var kernelHandle = shRenderShader.FindKernel("CSRenderEquirectangular");
        var width = targetRenderTexture.width;
        var height = targetRenderTexture.height;
        
        var shBuffer = new ComputeBuffer(SHCoefficientGenerator.SH_COEFFICIENT_COUNT, sizeof(float) * 3);
        shBuffer.SetData(shCoefficients);
        
        shRenderShader.SetBuffer(kernelHandle, "_SHCoefficients", shBuffer);
        shRenderShader.SetInts("_Resolution", width, height);
        shRenderShader.SetTexture(kernelHandle, "_OutputEquirectangular", targetRenderTexture);

        var threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        var threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        shRenderShader.Dispatch(kernelHandle, threadGroupsX, threadGroupsY, 1);
        
        shBuffer.Release();
    }

    private void Update()
    {
        if (shCoefficients == null) { return; }

        RenderToTexture();
    }
}
