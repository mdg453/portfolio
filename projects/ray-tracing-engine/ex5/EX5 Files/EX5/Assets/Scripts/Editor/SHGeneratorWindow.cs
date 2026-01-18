using System.IO;
using UnityEngine;
using UnityEditor;

public class SHGeneratorWindow : EditorWindow
{
    private Cubemap sourceCubemap;
    private ComputeShader shRenderShader;
    
    private int sampleCount = 100000;
    
    private Vector3[] shCoefficients;
    private string coefficientsString = "";

    [MenuItem("Tools/SH Generator")]
    public static void ShowWindow()
    {
        GetWindow<SHGeneratorWindow>("SH Generator");
    }

    private ComputeShader FindShader(string name)
    {
        var guids = AssetDatabase.FindAssets($"{name} t:ComputeShader");
        
        if (guids.Length > 0)
        {
            var path = AssetDatabase.GUIDToAssetPath(guids[0]);
            return AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
        }
        return null;
    }

    void OnGUI()
    {
        GUILayout.Label("Spherical Harmonics Generator (Compute)", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox("Step 1: Generate SH coefficients from a source cubemap.\nStep 2: Render a diffuse irradiance map from those coefficients.", MessageType.Info);
        
        EditorGUILayout.Space();
        
        // --- Step 1: Generation ---
        GUILayout.Label("Step 1: Generate Coefficients", EditorStyles.boldLabel);
        sourceCubemap = (Cubemap)EditorGUILayout.ObjectField("Source Cubemap", sourceCubemap, typeof(Cubemap), false);
        // sampleCount = EditorGUILayout.IntSlider("Sample Count", sampleCount, 1000, 5000000);

        if (GUILayout.Button("Generate SH Coefficients"))
        {
            if (ValidateAssets(true))
            {
                GenerateSHCoefficients();

                if (FindAnyObjectByType<SHVisualizer>() is { } shVisualizer)
                {
                    shVisualizer.SetSHCoefficients(shCoefficients);
                }

                if (FindAnyObjectByType<Part2SHSetter>() is { } shSetter)
                {
                    shSetter.SetSHCoefficients(shCoefficients);
                }
            }
        }

        EditorGUILayout.Space();

        // --- Display Coefficients ---
        GUILayout.Label("Generated Coefficients (L0 to L3)", EditorStyles.boldLabel);
        EditorGUILayout.TextArea(coefficientsString, GUILayout.Height(150));
        
        EditorGUILayout.Space();
        
        // --- Step 2: Rendering ---
        GUILayout.Label("Step 2: Render Irradiance Map", EditorStyles.boldLabel);

        // Disable button if we don't have coefficients yet.
        GUI.enabled = shCoefficients != null && shCoefficients.Length > 0;

        if (GUILayout.Button("Save Irradiance Map"))
        {
            if (ValidateAssets(false))
            {
                GenerateEquirectangularMap();
            }
        }
        GUI.enabled = true;
    }

    private bool ValidateAssets(bool isGenerating)
    {
        if (isGenerating)
        {
            if (sourceCubemap == null) { EditorUtility.DisplayDialog("Error", "Please assign a source cubemap.", "OK"); return false; }
        }
        else // isRendering
        {
            if (shRenderShader == null) { EditorUtility.DisplayDialog("Error", "Please assign the SHRenderer compute shader.", "OK"); return false; }
            if (shCoefficients == null || shCoefficients.Length == 0) { EditorUtility.DisplayDialog("Error", "No SH coefficients have been generated yet.", "OK"); return false; }
        }
        return true;
    }
    
    private void GenerateSHCoefficients()
    {
        // Call the new static helper class to do the work.
        shCoefficients = SHCoefficientGenerator.GenerateCoefficients(sourceCubemap, sampleCount);
        FormatCoefficientsString();
    }
    
    private void GenerateEquirectangularMap()
    {
        const int outputWidth = 512;
        
        var path = EditorUtility.SaveFilePanel("Save Equirectangular Irradiance Map", "Assets", sourceCubemap.name + "_Irradiance", "exr");
        
        if (string.IsNullOrEmpty(path)) { return; }
        
        var kernelHandle = shRenderShader.FindKernel("CSRenderEquirectangular");
        var height = outputWidth / 2;
        var rt = new RenderTexture(outputWidth, height, 0, RenderTextureFormat.ARGBFloat) { enableRandomWrite = true };
        rt.Create();

        // Buffer size is now 16 * sizeof(Vector3)
        var shBuffer = new ComputeBuffer(SHCoefficientGenerator.SH_COEFFICIENT_COUNT, sizeof(float) * 3);
        shBuffer.SetData(shCoefficients);
        
        shRenderShader.SetBuffer(kernelHandle, "_SHCoefficients", shBuffer);
        shRenderShader.SetInts("_Resolution", outputWidth, height);
        shRenderShader.SetTexture(kernelHandle, "_OutputEquirectangular", rt);

        var threadGroupsX = Mathf.CeilToInt(outputWidth / 8.0f);
        var threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        shRenderShader.Dispatch(kernelHandle, threadGroupsX, threadGroupsY, 1);

        var outputTex = new Texture2D(outputWidth, height, TextureFormat.RGBAFloat, false);
        RenderTexture.active = rt;
        outputTex.ReadPixels(new Rect(0, 0, outputWidth, height), 0, 0);
        outputTex.Apply();
        RenderTexture.active = null;
        
        var bytes = outputTex.EncodeToEXR(Texture2D.EXRFlags.CompressZIP);
        File.WriteAllBytes(path, bytes);
        
        DestroyImmediate(outputTex);
        rt.Release();
        shBuffer.Release();
        AssetDatabase.Refresh();
        Debug.Log($"Successfully saved HDR irradiance map to: {path}");
    }

    private void FormatCoefficientsString()
    {
        if (shCoefficients == null) { return; }
        
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("// Spherical Harmonics Coefficients (L0 to L3)");
        
        for (var i = 0; i < shCoefficients.Length; i++)
        {
            string bandLabel;
            
            if (i < 1) bandLabel = "L0";
            else if (i < 4) bandLabel = "L1";
            else if (i < 9) bandLabel = "L2";
            else bandLabel = "L3";
            
            var c = shCoefficients[i];
            sb.AppendLine($"({c.x:F3}f, {c.y:F3}f, {c.z:F3}f), // {bandLabel}");
        }
        
        coefficientsString = sb.ToString();
    }
}