import { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

// Earth component
const Earth = () => {
  const earthRef = useRef();
  const cloudRef = useRef();
  const [texturesLoaded, setTexturesLoaded] = useState(false);
  
  // Load textures with error handling
  const [colorMap, normalMap, specularMap, bumpMap] = useLoader(THREE.TextureLoader, [
    'https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg',
    'https://threejs.org/examples/textures/planets/earth_normal_2048.jpg',
    'https://threejs.org/examples/textures/planets/earth_specular_2048.jpg',
    'https://threejs.org/examples/textures/planets/earth_normal_2048.jpg' // Using normal map as bump map for simplicity
  ], (loader) => {
    // Set loaded to true when all textures are loaded
    setTexturesLoaded(true);
  });
  
  useFrame(() => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.001;
    }
    if (cloudRef.current) {
      cloudRef.current.rotation.y += 0.0005;
    }
  });

  return (
    <>
      <ambientLight intensity={0.4} color="#ffffff" />
      <directionalLight 
        position={[5, 3, 5]} 
        intensity={2} 
        color="#ffffff"
        castShadow
      />
      <pointLight position={[-10, -5, -5]} intensity={0.8} color="#4466ff" />
      
      <Sphere ref={earthRef} args={[2, 64, 64]}>
        {texturesLoaded ? (
          <meshPhongMaterial
            map={colorMap}
            normalMap={normalMap}
            specularMap={specularMap}
            bumpMap={bumpMap}
            bumpScale={1}
            specular={new THREE.Color(0x444444)}
            shininess={25}
            wireframe={false}
          />
        ) : (
          <meshPhongMaterial
            color="#1a4d7a"
            specular="#222222"
            shininess={10}
          />
        )}
      </Sphere>
      {texturesLoaded && (
        <Sphere ref={cloudRef} args={[2.02, 64, 64]}>
          <meshPhongMaterial
            map={useLoader(THREE.TextureLoader, 'https://threejs.org/examples/textures/planets/earth_clouds_1024.png')}
            transparent={true}
            opacity={0.7}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </Sphere>
      )}
    </>
  );
};

// Satellite marker with horizontal orbital motion
const SatelliteMarker = ({ position, color, onClick, isSelected, index }) => {
  const [hovered, setHovered] = useState(false);
  const meshRef = useRef();
  
  // Fixed orbital parameters for 5 satellites
  const orbitRadius = 3.2; // Slightly larger radius for better visibility
  const orbitSpeed = 0.04; // Slower speed for better tracking
  const orbitOffset = (index / 5) * Math.PI * 2; // Evenly space satellites
  
  useFrame(({ clock }) => {
    if (meshRef.current) {
      const time = clock.getElapsedTime() * orbitSpeed;
      const angle = time + orbitOffset;
      
      // Calculate position in horizontal plane (y = 0 for all points)
      const x = Math.cos(angle) * orbitRadius;
      const z = Math.sin(angle) * orbitRadius;
      
      meshRef.current.position.set(x, 0, z);
      
      // Make satellite face direction of travel (in horizontal plane)
      const lookAtX = Math.cos(angle + 0.1) * orbitRadius;
      const lookAtZ = Math.sin(angle + 0.1) * orbitRadius;
      meshRef.current.lookAt(lookAtX, 0, lookAtZ);
    }
  });

  return (
    <mesh
      ref={meshRef}
      onClick={onClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[isSelected || hovered ? 0.08 : 0.06, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isSelected || hovered ? 1 : 0.5}
      />
      {(isSelected || hovered) && (
        <pointLight color={color} intensity={1} distance={2} />
      )}
    </mesh>
  );
};

// Orbit rings
const OrbitRings = () => {
  const innerPoints = [];
  const outerPoints = [];
  const segments = 100;
  
  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    innerPoints.push(new THREE.Vector3(
      Math.cos(theta) * 2.3,
      0,
      Math.sin(theta) * 2.3
    ));
    outerPoints.push(new THREE.Vector3(
      Math.cos(theta) * 3.5,
      0,
      Math.sin(theta) * 3.5
    ));
  }

  return (
    <>
      <Line points={innerPoints} color="#1a3a52" lineWidth={1} />
      <Line points={outerPoints} color="#1a3a52" lineWidth={1} />
    </>
  );
};

const GlobeView = ({ satellites, selectedSatellite, onSelectSatellite }) => {
  if (!satellites) return null;

  return (
    <div className="bg-hud-darker border border-hud-border rounded-lg p-4 h-[600px]">
      <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
        <ambientLight intensity={0.4} color="#ffffff" />
        <directionalLight 
          position={[5, 3, 5]} 
          intensity={2} 
          color="#ffffff"
          castShadow
        />
        <pointLight position={[-10, -5, -5]} intensity={0.8} color="#4466ff" />
        
        <Earth />
        <OrbitRings />
        
        {satellites.map((sat, index) => (
          <SatelliteMarker
            key={index}
            index={index}
            position={sat.position}
            color={sat.status_color === 'green' ? '#00ff88' : '#ff4757'}
            isSelected={selectedSatellite?.id === sat.id}
            onClick={(e) => {
              e.stopPropagation();
              onSelectSatellite(sat);
            }}
          />
        ))}
        
        <OrbitControls
          enableZoom={true}
          enablePan={false}
          enableRotate={true}
          zoomSpeed={0.6}
          rotateSpeed={0.4}
          minDistance={5}
          maxDistance={12}
        />
      </Canvas>
      
      {/* Satellite Info Card */}
      {selectedSatellite && (
        <div className="absolute bottom-8 left-8 bg-hud-darker border border-hud-accent rounded-lg p-4 w-64 shadow-hud">
          <div className="flex items-start justify-between mb-3">
            <div>
              <div className={`inline-flex items-center px-2 py-1 rounded text-xs font-semibold ${
                selectedSatellite.status_color === 'red' 
                  ? 'bg-hud-red/20 text-hud-red' 
                  : 'bg-hud-green/20 text-hud-green'
              }`}>
                ● {selectedSatellite.id}
              </div>
            </div>
            <button
              onClick={() => onSelectSatellite(null)}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Type:</span>
              <span className="text-white font-medium">{selectedSatellite.type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Clock Error:</span>
              <span className="text-hud-accent">{selectedSatellite.error_data.clock_error}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Ephemeris Error:</span>
              <span className="text-hud-accent">{selectedSatellite.error_data.ephemeris_error}</span>
            </div>
          </div>
          
          <button className="w-full mt-4 bg-hud-accent/20 hover:bg-hud-accent/30 text-hud-accent py-2 rounded text-sm font-medium transition-colors">
            Details
          </button>
        </div>
      )}
    </div>
  );
};

export default GlobeView;
