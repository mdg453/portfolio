// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            Blend OneMinusDstAlpha One
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma require compute
            #pragma use_dxc

            #include "GaussianSplatting.hlsl"

            StructuredBuffer<SplatViewData> _SplatViewData;
            StructuredBuffer<uint> _OrderBuffer;
            ByteAddressBuffer _SplatSelectedBits;
            
            float4 _BoundsMin;
            float4 _BoundsMax;

			uint _SplatBitsValid;

			// Add floating point epsilon to avoid NaNs
			static const float EPSILON = 1e-5;

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 pos : TEXCOORD0;
                float4 col : TEXCOORD1;
				float3 center : TEXCOORD2;
            };

			// Animation parameters (must match fragment shader)
		static const float ANIM_PORTAL_SPEED = 0.15;      // Same as fragment shader
		static const float ANIM_PULSE_WINDOW = 0.15;     // Window size for pulsation effect
		static const float ANIM_PULSE_STRENGTH = 0.08;   // How far splats move outward
		static const float ANIM_PULSE_FREQ = 8.0;        // Frequency of pulsation

		float3 animatePosition(float3 center) {
			// Normalize Y position to [0, 1] range using bounds
			float normalizedY = (center.y - _BoundsMin.y) / (_BoundsMax.y - _BoundsMin.y + EPSILON);
			
			// Animate portal position: sweeps from top (1.0) to bottom (0.0)
			float portalY = 1.0 - frac(_Time.y * ANIM_PORTAL_SPEED);
			
			// Distance from portal plane
			float distFromPortal = normalizedY - portalY;
			
			// Only affect splats near the portal edge (within the rolling window)
			// Splats below portal or far above are not affected
			if (distFromPortal < 0 || distFromPortal > ANIM_PULSE_WINDOW) {
				return center;
			}
			
			// Calculate pulsation factor using smoothstep for smooth transition
			// Strongest pulsation right at edge, fading as we move away
			float pulseFactor = 1.0 - smoothstep(0.0, ANIM_PULSE_WINDOW, distFromPortal);
			
			// Calculate outward direction in XZ plane (radial from center)
			float2 xzDir = normalize(center.xz + float2(EPSILON, EPSILON));
			
			// Pulsating displacement using sin wave for oscillation
			float pulse = sin(_Time.y * ANIM_PULSE_FREQ) * pulseFactor * ANIM_PULSE_STRENGTH;
			
			// Apply outward displacement in XZ plane
			float3 newPos = center;
			newPos.xz += xzDir * pulse;
			
			return newPos;
		}	

			v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
			{
			    v2f o = (v2f)0;
			    instID = _OrderBuffer[instID];
				SplatViewData view = _SplatViewData[instID];

				float3 newCenterPos = animatePosition(view.pos.xyz);
				o.center = newCenterPos;
				
				float3 centerWorldPos = mul(unity_ObjectToWorld, float4(newCenterPos,1)).xyz;
				float4 centerClipPos = mul(UNITY_MATRIX_VP, float4(centerWorldPos, 1));
				
				bool behindCam = centerClipPos.w <= 0;
				
				if (behindCam)
				{
					o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
				}
				else
				{
					o.col.r = f16tof32(view.color.x >> 16);
					o.col.g = f16tof32(view.color.x);
					o.col.b = f16tof32(view.color.y >> 16);
					o.col.a = f16tof32(view.color.y);

					uint idx = vtxID;
					float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
					quadPos *= 2;

					o.pos = quadPos;

					float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2 / _ScreenParams.xy;
					o.vertex = centerClipPos;
					float cameraDist = centerClipPos.w;
					
					o.vertex.xy += deltaScreenPos * cameraDist;

					// is this splat selected?
					if (_SplatBitsValid)
					{
						uint wordIdx = instID / 32;
						uint bitIdx = instID & 31;
						uint selVal = _SplatSelectedBits.Load(wordIdx * 4);

						if (selVal & (1 << bitIdx))
						{
							o.col.a = -1;				
						}
					}
				}
				FlipProjectionIfBackbuffer(o.vertex);
			    return o;
			}

			struct effect_result
		{
			half3 effectColor;
			float alpha;
		};

		// Portal effect parameters
		static const float PORTAL_SPEED = 0.15;           // Speed of portal sweep (cycles per second)
		static const float PORTAL_THICKNESS = 0.1;       // Thickness of the glowing edge
		static const half3 PORTAL_GLOW_COLOR = half3(30.0, 5.0, 0.0); // HDR red/orange glow

		effect_result calculateEffectColor(float3 center, float alpha, half3 color)
		{
			effect_result output;
			output.effectColor = color;
			output.alpha = alpha;
			
			// Normalize Y position to [0, 1] range using bounds
			float normalizedY = (center.y - _BoundsMin.y) / (_BoundsMax.y - _BoundsMin.y + EPSILON);
			
			// Animate portal position: sweeps from top (1.0) to bottom (0.0) over time
			// frac creates a repeating cycle
			float portalY = 1.0 - frac(_Time.y * PORTAL_SPEED);
			
			// Calculate distance from portal plane
			float distFromPortal = normalizedY - portalY;
			
			// Hide splats below the portal (creates reveal effect from top to bottom)
			if (distFromPortal < 0)
			{
				output.alpha = 0;
				return output;
			}
			
			// Create smooth glow at the edge using smoothstep
			// Glow is strongest right at the portal edge and fades out above
			float glowFactor = 1.0 - smoothstep(0.0, PORTAL_THICKNESS, distFromPortal);
			
			// Lerp between original color and HDR glow color
			output.effectColor = lerp(color, PORTAL_GLOW_COLOR, glowFactor);
			
			// Boost alpha at the edge for more visible glow
			output.alpha = lerp(alpha, min(alpha * 2.0, 1.0), glowFactor);
			
			return output;
		}

            half4 frag (v2f i) : SV_Target
            {
				// Calculate Gaussian falloff based on distance from center
				// i.pos is the quad position in [-2, 2] range
				// Following standard Gaussian splatting: alpha = exp(-0.5 * r^2)
				float power = -0.5 * dot(i.pos, i.pos);
				float gaussianWeight = exp(power);
				
				// Clip fragments with very low alpha for efficiency
				if (gaussianWeight < 1.0/255.0)
					discard;
				
				// Combine Gaussian weight with splat's base opacity
				float alpha = gaussianWeight * saturate(i.col.a);
				
				// If splat is selected (negative alpha), highlight it
				if (i.col.a < 0)
					alpha = gaussianWeight;
				
				// Apply effect color calculation
				effect_result effect = calculateEffectColor(i.center, alpha, i.col.rgb);
				
				// Premultiplied alpha output for back-to-front blending
                return half4(effect.effectColor * effect.alpha, effect.alpha);
            }
            ENDCG
        }
    }
}
