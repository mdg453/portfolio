using UnityEngine;

[ExecuteAlways]
public class Part2SHSetter : MonoBehaviour
{
    [SerializeField] private ComputeShader shRenderShader;
    [SerializeField] private RenderTexture targetRenderTexture;
    
    [Range(0, 1)] [SerializeField] private float roughness = 0.5f;
    [Range(0, 1)] [SerializeField] private float metallic = 1.0f;
    [SerializeField] private Cubemap envTex;

    private ComputeBuffer shBuffer;
    private int kernelHandle;

    void OnEnable()
    {
        shBuffer = new ComputeBuffer(SHCoefficientGenerator.SH_COEFFICIENT_COUNT, sizeof(float) * 3);
        kernelHandle = shRenderShader.FindKernel("CSRaymarch");
        SetDefaultBuffer();
    }

    private void Update()
    {
        UpdateShaderParameters();
        DispatchShader();
    }

    private void UpdateShaderParameters()
    {
        Camera cam = Camera.main;

        // Matrix Math for Ray Reconstruction
        // We need the Inverse View (Camera to World) and Inverse Projection (Clip to View)
        Matrix4x4 invView = cam.cameraToWorldMatrix;
        Matrix4x4 invProj = cam.projectionMatrix.inverse;

        shRenderShader.SetMatrix("_InvViewMatrix", invView);
        shRenderShader.SetMatrix("_InvProjectionMatrix", invProj);
        shRenderShader.SetVector("_WorldSpaceCameraPos", cam.transform.position);
        shRenderShader.SetVector("_Time", new Vector4(Time.time / 20, Time.time, Time.time * 2, Time.time * 3));
        
        shRenderShader.SetFloat("_Roughness", roughness);
        shRenderShader.SetFloat("_Metallic", metallic);
        
        shRenderShader.SetTexture(kernelHandle, "_EnvTex", envTex);
        
        shRenderShader.SetTexture(kernelHandle, "Result", targetRenderTexture);
        shRenderShader.SetBuffer(kernelHandle, "_SHCoefficients", shBuffer);
    }

    private void DispatchShader()
    {
        // Calculate thread groups based on the [numthreads(8, 8, 1)] in the shader
        int groupsX = Mathf.CeilToInt(targetRenderTexture.width / 8.0f);
        int groupsY = Mathf.CeilToInt(targetRenderTexture.height / 8.0f);
        
        shRenderShader.Dispatch(kernelHandle, groupsX, groupsY, 1);
    }

    private void SetDefaultBuffer()
    {
        if (shBuffer == null) { return; }
        shBuffer.SetData(new Vector3[SHCoefficientGenerator.SH_COEFFICIENT_COUNT]);
    }

    public void SetSHCoefficients(Vector3[] shCoefficients)
    {
        if (shBuffer != null && shCoefficients != null)
        {
            shBuffer.SetData(shCoefficients);
        }
    }

    void OnDisable()
    {
        if (shBuffer != null)
        {
            shBuffer.Release();
            shBuffer = null;
        }
    }
}