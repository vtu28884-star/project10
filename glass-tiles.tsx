import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

const vertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const fragmentShader = `
precision highp float;

varying vec2 vUv;

uniform float uTime;
uniform vec2  uRes;
uniform float uSpeed;
uniform float uDensity;
uniform float uLayers;
uniform float uWarp;
uniform float uBand;
uniform float uChromatic;
uniform vec3  uColorA;
uniform vec3  uColorB;
uniform vec3  uBg;
uniform float uAlpha;

const int   MAX_LAYERS    = 8;
const float GOLDEN_ANGLE  = 2.39996323;

float sampleField(vec2 p, float t) {
  vec2 c = p;
  vec2 s = floor(c);
  vec2 pole = vec2(uWarp) / (s - c + 1e-3);

  for (int j = 1; j <= MAX_LAYERS; j++) {
    if (float(j) > uLayers) break;
    float fi = float(j);
    float ang = GOLDEN_ANGLE * fi;
    vec2 dir = vec2(cos(ang), sin(ang));
    float detune = 1.0 + fi * 0.13;
    float phase = dot(c, dir) * detune + pole.x + pole.y + t;
    c += dir * sin(phase) / fi;
  }

  float band = exp(-max(uBand, 0.0001) * abs(sin(c.y)));
  return band;
}

void main() {
  vec2 fragPx = vUv * uRes;
  vec2 base = fragPx / max(uRes.y, 1.0) * uDensity + uRes / max(uRes.y, 1.0);

  float t = uTime * uSpeed;

  float spread = uChromatic * 0.35;
  float r = sampleField(base + vec2( spread, 0.0), t);
  float g = sampleField(base,                      t);
  float b = sampleField(base + vec2(-spread, 0.0), t);

  vec3 col = vec3(
    mix(uColorA.r, uColorB.r, r),
    mix(uColorA.g, uColorB.g, g),
    mix(uColorA.b, uColorB.b, b)
  );

  float fieldMix = clamp(max(max(r, g), b), 0.0, 1.0);
  vec3 final = mix(uBg, col, fieldMix);

  gl_FragColor = vec4(final, uAlpha);
}
`;

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16) / 255, parseInt(result[2], 16) / 255, parseInt(result[3], 16) / 255]
    : [0, 0, 0];
}

interface GlassTilesMeshProps {
  speed?: number;
  tileDensity?: number;
  rippleLayers?: number;
  warpStrength?: number;
  bandSharpness?: number;
  chromaticSpread?: number;
  colorA?: string;
  colorB?: string;
  backgroundColor?: string;
  opacity?: number;
}

const GlassTilesMesh: React.FC<GlassTilesMeshProps> = ({
  speed = 1,
  tileDensity = 4,
  rippleLayers = 6,
  warpStrength = 0.33,
  bandSharpness = 3,
  chromaticSpread = 0,
  colorA = '#1E00FF',
  colorB = '#D765E6',
  backgroundColor = '#000000',
  opacity = 1
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { size } = useThree();

  useFrame((state) => {
    if (!meshRef.current) return;
    const material = meshRef.current.material as THREE.ShaderMaterial;
    const uniforms = material.uniforms;
    uniforms.uTime.value = state.clock.elapsedTime;
    uniforms.uRes.value.set(size.width, size.height);
    uniforms.uSpeed.value = speed;
    uniforms.uDensity.value = tileDensity;
    uniforms.uLayers.value = Math.max(1, Math.min(8, Math.round(rippleLayers)));
    uniforms.uWarp.value = warpStrength;
    uniforms.uBand.value = bandSharpness;
    uniforms.uChromatic.value = chromaticSpread;
    uniforms.uAlpha.value = opacity;

    const cA = hexToRgb(colorA);
    uniforms.uColorA.value.set(cA[0], cA[1], cA[2]);
    const cB = hexToRgb(colorB);
    uniforms.uColorB.value.set(cB[0], cB[1], cB[2]);
    const bg = hexToRgb(backgroundColor);
    uniforms.uBg.value.set(bg[0], bg[1], bg[2]);
  });

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uRes: { value: new THREE.Vector2(1, 1) },
      uSpeed: { value: 1 },
      uDensity: { value: 4 },
      uLayers: { value: 5 },
      uWarp: { value: 0.1 },
      uBand: { value: 3 },
      uChromatic: { value: 0.25 },
      uColorA: { value: new THREE.Color(0.05, 0.02, 0.1) },
      uColorB: { value: new THREE.Color(0.4, 0.5, 0.9) },
      uBg: { value: new THREE.Color(0, 0, 0) },
      uAlpha: { value: 1 }
    }),
    []
  );

  return (
    <mesh ref={meshRef}>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
      />
    </mesh>
  );
};

export interface GlassTilesProps {
  width?: string | number;
  height?: string | number;
  className?: string;
  children?: React.ReactNode;
  speed?: number;
  tileDensity?: number;
  rippleLayers?: number;
  warpStrength?: number;
  bandSharpness?: number;
  chromaticSpread?: number;
  colorA?: string;
  colorB?: string;
  backgroundColor?: string;
  opacity?: number;
  dpr?: number;
}

export const GlassTiles: React.FC<GlassTilesProps> = ({
  width = '100%',
  height = '100%',
  className = '',
  children,
  speed = 1,
  tileDensity = 4,
  rippleLayers = 6,
  warpStrength = 0.33,
  bandSharpness = 3,
  chromaticSpread = 0,
  colorA = '#1E00FF',
  colorB = '#D765E6',
  backgroundColor = '#000000',
  opacity = 1,
  dpr = 1.5
}) => {
  return (
    <div className={`relative overflow-hidden ${className}`} style={{ width, height }}>
      <Canvas
        className="absolute inset-0"
        dpr={[1, dpr]}
        gl={{ antialias: false, alpha: true, powerPreference: 'high-performance' }}
        orthographic
        camera={{ position: [0, 0, 1], zoom: 1, left: -1, right: 1, top: 1, bottom: -1 }}
      >
        <GlassTilesMesh
          speed={speed}
          tileDensity={tileDensity}
          rippleLayers={rippleLayers}
          warpStrength={warpStrength}
          bandSharpness={bandSharpness}
          chromaticSpread={chromaticSpread}
          colorA={colorA}
          colorB={colorB}
          backgroundColor={backgroundColor}
          opacity={opacity}
        />
      </Canvas>
      {children && <div className="relative z-10">{children}</div>}
    </div>
  );
};

export default GlassTiles;

