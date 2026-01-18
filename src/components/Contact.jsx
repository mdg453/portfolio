
import React from 'react';
import { Mail, Phone, MapPin, Github, Linkedin } from "lucide-react";

const Contact = () => {
    return (
        <section id="contact" className="py-24 bg-muted/30">
            <div className="container mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold text-foreground mb-4">Get In Touch</h2>
                    <div className="w-16 h-1 bg-primary rounded-full mx-auto mb-4"></div>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        Have a project in mind or just want to say hi? I'd love to hear from you.
                    </p>
                </div>

                <div className="max-w-xl mx-auto">
                    <div className="bg-background p-10 rounded-2xl shadow-xl border border-border text-center">
                        <h3 className="text-2xl font-bold mb-8 text-foreground">Contact Information</h3>
                        <div className="flex flex-col gap-6 items-center">
                            <div className="flex items-center gap-4 text-lg text-muted-foreground hover:text-primary transition-colors">
                                <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                                    <Mail className="w-5 h-5" />
                                </div>
                                <a href="mailto:meitar453@gmail.com">meitar453@gmail.com</a>
                            </div>
                            <div className="flex items-center gap-4 text-lg text-muted-foreground hover:text-primary transition-colors">
                                <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                                    <Phone className="w-5 h-5" />
                                </div>
                                <a href="tel:0523808106">052-3808106</a>
                            </div>
                            <div className="flex items-center gap-4 text-lg text-muted-foreground">
                                <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                                    <MapPin className="w-5 h-5" />
                                </div>
                                <span>Jerusalem, Israel</span>
                            </div>
                        </div>

                        <div className="flex justify-center gap-6 mt-10">
                            <a
                                href="https://github.com/mdg453"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="w-12 h-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground hover:bg-primary hover:text-primary-foreground hover:-translate-y-1 transition-all shadow-sm"
                                aria-label="GitHub"
                            >
                                <Github className="w-6 h-6" />
                            </a>
                            <a
                                href="https://www.linkedin.com/in/meitar-de-gracea"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="w-12 h-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground hover:bg-primary hover:text-primary-foreground hover:-translate-y-1 transition-all shadow-sm"
                                aria-label="LinkedIn"
                            >
                                <Linkedin className="w-6 h-6" />
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Contact;
