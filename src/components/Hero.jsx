
import React from 'react';
import { ArrowRight, Mail } from "lucide-react";
import profileImg from '../assets/profile.jpg';

const Hero = () => {
    return (
        <section id="home" className="min-h-screen pt-20 flex items-center relative overflow-hidden bg-gradient-to-b from-background via-muted/30 to-background">
            {/* Background Accents */}
            <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] rounded-full bg-primary/5 blur-3xl -z-10"></div>
            <div className="absolute bottom-[-10%] left-[-5%] w-[500px] h-[500px] rounded-full bg-indigo-500/5 blur-3xl -z-10"></div>

            <div className="container mx-auto px-6 grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                {/* Text Content */}
                <div className="order-2 md:order-1 flex flex-col items-start animate-in slide-in-from-bottom-5 fade-in duration-700">
                    <span className="inline-block px-4 py-1.5 rounded-full bg-primary/10 text-primary font-medium text-sm mb-6">
                        EE & CS Student @ Hebrew University
                    </span>
                    <h1 className="text-4xl md:text-6xl font-bold leading-tight text-foreground mb-6">
                        Specializing in <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-indigo-600">Verification Automation</span> & Embedded Systems.
                    </h1>
                    <p className="text-lg text-muted-foreground mb-8 max-w-lg">
                        I'm Meitar, a developer with deep expertise in Hardware-Software integration, Firmware Validation,
                        and building automated test environments for constrained embedded systems.
                    </p>
                    <div className="flex flex-wrap gap-4">
                        <a href="#projects" className="inline-flex items-center gap-2 bg-foreground text-background px-8 py-3 rounded-full font-medium hover:bg-foreground/90 transition-transform hover:scale-105">
                            View Projects
                            <ArrowRight className="w-4 h-4" />
                        </a>
                        <a href="#contact" className="inline-flex items-center gap-2 bg-background border border-border text-foreground px-8 py-3 rounded-full font-medium hover:bg-muted transition-colors">
                            <Mail className="w-4 h-4" />
                            Contact Me
                        </a>
                    </div>
                </div>

                {/* Image Content */}
                <div className="order-1 md:order-2 flex justify-center md:justify-end animate-in slide-in-from-right-5 fade-in duration-1000">
                    <div className="relative w-[300px] h-[300px] md:w-[450px] md:h-[450px]">
                        {/* Spinning Rings */}
                        <div className="absolute inset-0 rounded-full border border-border/60 animate-[spin_30s_linear_infinite]"></div>
                        <div className="absolute inset-4 rounded-full border border-primary/20 animate-[spin_20s_linear_infinite_reverse]"></div>

                        {/* Profile Image */}
                        <div className="absolute inset-0 rounded-full overflow-hidden shadow-2xl border-4 border-background">
                            <img
                                src={profileImg}
                                alt="Meitar"
                                className="w-full h-full object-cover object-top"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Hero;
