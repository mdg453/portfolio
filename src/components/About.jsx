
import React from 'react';
import workspaceImg from '../assets/workspace.jpg';

const About = () => {
    return (
        <section id="about" className="py-24 bg-background">
            <div className="container mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold text-foreground mb-4">About Me</h2>
                    <div className="w-16 h-1 bg-primary rounded-full mx-auto mb-4"></div>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        Here's a little bit about me and my journey.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                    <div className="relative group">
                        <div className="absolute inset-0 bg-primary rounded-2xl md:rotate-6 opacity-20 blur-lg transition-transform group-hover:rotate-3"></div>
                        <img
                            src={workspaceImg}
                            alt="Workspace"
                            className="relative rounded-2xl shadow-xl w-full object-cover transform transition-transform group-hover:scale-[1.01]"
                        />
                    </div>

                    <div className="flex flex-col">
                        <h3 className="text-2xl font-semibold mb-6">Verification Automation Engineer & Developer.</h3>
                        <p className="text-muted-foreground mb-6 leading-relaxed">
                            I am a final-year Electrical Engineering and Computer Science student with nearly 2 years of
                            experience in Verification Automation at Mobileye. My work focuses on Hardware-Software
                            integration, system-level debugging, and developing robust infrastructure for firmware
                            validation.
                        </p>
                        <p className="text-muted-foreground mb-8 leading-relaxed">
                            My technical toolkit includes advanced proficiency in Python and C++, along with hands-on
                            experience in Embedded Linux, Computer Architecture, and DevOps tools like Docker, Kubernetes,
                            and Jenkins.
                        </p>

                        <div className="grid grid-cols-3 gap-6">
                            {[
                                { number: "2+", label: "Years Experience" },
                                { number: "20+", label: "Projects Completed" },
                                { number: "EE/CS", label: "Double Major" }
                            ].map((stat, idx) => (
                                <div key={idx} className="bg-muted/50 p-4 rounded-xl text-center hover:bg-muted transition-colors">
                                    <h4 className="text-3xl font-bold text-primary mb-1">{stat.number}</h4>
                                    <p className="text-sm text-muted-foreground font-medium">{stat.label}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default About;
