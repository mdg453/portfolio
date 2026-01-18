
import hebrewTranscriberImg from '../assets/hebrew_transcriber.png';
import panoramaImg from '../assets/panorama.png';
import pyramidBlendingImg from '../assets/thumb_pyramid_blending.png';
import sceneCutImg from '../assets/thumb_scene_cut_detector.png';
import audioWatermarkingImg from '../assets/thumb_audio_watermarking.png';
import palindromeImg from '../assets/palindrome.png';
import graphicsImg from '../assets/graphics.png';
import converterImg from '../assets/converter.png';
import gptutorImg from '../assets/gptutor.png';
import audioImg from '../assets/audio.png';
import fllImg from '../assets/fll.png';
import detectagasImg from '../assets/detectagas.png';
import cppImg from '../assets/cpp.png';
import oopImg from '../assets/oop.png';

export const projects = [
    {
        title: "Hebrew Transcriber",
        description: "High-accuracy Hebrew speech-to-text tool using fine-tuned Whisper models and FFmpeg.",
        tags: ["Python", "Whisper AI", "FFmpeg"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/hebrew-transcriber",
        image: hebrewTranscriberImg
    },
    {
        title: "Panorama Stitching",
        description: "Advanced video processing algorithms to create seamless high-resolution panoramas.",
        tags: ["Computer Vision", "Python", "NumPy"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/panorama-stitching",
        image: panoramaImg
    },
    {
        title: "Pyramid Blending & Hybrid Images",
        description: "Seamless image blending and hybrid image construction using Gaussian and Laplacian pyramids.",
        tags: ["Python", "OpenCV", "Computer Vision"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/pyramid-blending",
        image: pyramidBlendingImg
    },
    {
        title: "Video Scene Cut Detector",
        description: "Minimal pure-Python tool for detecting scene transitions in video streams using histograms.",
        tags: ["Python", "FFmpeg", "Video Processing"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/video-scene-cut-detector",
        image: sceneCutImg
    },
    {
        title: "Audio Watermarking Utility",
        description: "Steganographic tool for embedding inaudible spread-spectrum watermarks into audio files.",
        tags: ["Python", "SciPy", "Signal Processing"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/audio-watermarking",
        image: audioWatermarkingImg
    },
    {
        title: "Palindrome API",
        description: "FastAPI web service for efficient palindrome detection using Manacher's Algorithm.",
        tags: ["Python", "FastAPI", "Docker"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/palindrome-api",
        image: palindromeImg
    },
    {
        title: "3D Ray Tracing Engine",
        description: "Physically-based rendering engine supporting shadows, reflections, and refractions.",
        tags: ["C++", "Computer Graphics", "Rendering"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/ray-tracing-engine",
        image: graphicsImg
    },
    {
        title: "SevenConverter",
        description: "Robust video/audio conversion tool wrapper for FFMPEG with multi-threading.",
        tags: ["C#/.NET", "FFMPEG", "Desktop App"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/seven-converter",
        image: converterImg
    },
    {
        title: "gpTutor",
        description: "Hackathon project: generative AI tutor for personalized learning experiences.",
        tags: ["Python", "Generative AI", "Hackathon"],
        link: "https://github.com/mdg453/gpTutor",
        image: gptutorImg
    },
    {
        title: "Audio Signal Processing",
        description: "Advanced noise reduction and spectral analysis algorithms for audio enhancement.",
        tags: ["MATLAB", "Signal Processing", "Algorithms"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/audio-signal-processing",
        image: audioImg
    },
    {
        title: "FLL Robotics Controller",
        description: "Pybricks Python control software for competitive Lego Spike Prime robots.",
        tags: ["Python", "Robotics", "Pybricks"],
        link: "https://github.com/mdg453/portfolio/tree/main/projects/fll-robotics",
        image: fllImg
    },
    {
        title: "DetectaGas",
        description: "Google Developer Student Club (GDSC) project for gas detection and safety.",
        tags: ["C++", "IoT", "GDSC"],
        link: "https://github.com/mdg453/DetectaGas",
        image: detectagasImg
    },
    {
        title: "C++ Programming",
        description: "Advanced C++ programming exercises focusing on memory management and efficiency.",
        tags: ["C++", "Algorithms", "Performance"],
        link: "https://github.com/mdg453/cpp---1",
        image: cppImg
    },
    {
        title: "OOP Final Project",
        description: "Comprehensive Object-Oriented Programming final assignment showcasing Java mastery.",
        tags: ["Java", "Design Patterns", "University"],
        link: "https://github.com/mdg453/oopex5_v2",
        image: oopImg
    }
];
