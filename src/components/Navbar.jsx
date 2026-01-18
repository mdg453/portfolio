import React, { useState, useEffect } from 'react';
import { cn } from "@/lib/utils";
import { Menu, X, Github } from "lucide-react";
import { ModeToggle } from "@/components/mode-toggle";

const Navbar = () => {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 50);
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navLinks = [
        { name: 'About', href: '#about' },
        { name: 'Projects', href: '#projects' },
        { name: 'Skills', href: '#skills' },
        { name: 'Contact', href: '#contact' },
    ];

    return (
        <>
            <nav
                className={cn(
                    "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
                    isScrolled
                        ? "bg-background/80 backdrop-blur-md border-b border-border/40 shadow-sm py-2"
                        : "bg-transparent py-4"
                )}
            >
                <div className="container mx-auto px-6 h-12 flex items-center justify-between">
                    <a href="#" className="text-2xl font-bold text-foreground">
                        Meitar<span className="text-primary">.</span>
                    </a>

                    {/* Desktop Nav */}
                    <div className="hidden md:flex items-center gap-8">
                        {navLinks.map((link) => (
                            <a
                                key={link.name}
                                href={link.href}
                                className="text-muted-foreground font-medium hover:text-primary transition-colors relative group"
                            >
                                {link.name}
                                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary transition-all group-hover:w-full"></span>
                            </a>
                        ))}
                        <a
                            href="https://github.com/mdg453"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-full text-sm font-medium hover:bg-primary/90 transition-colors"
                        >
                            <Github className="w-4 h-4" />
                            GitHub
                        </a>
                        <ModeToggle />
                    </div>

                    {/* Mobile Toggle */}
                    <div className="md:hidden flex items-center gap-4">
                        <ModeToggle />
                        <button
                            className="p-2 text-foreground"
                            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                            aria-label="Toggle menu"
                        >
                            {isMobileMenuOpen ? <X /> : <Menu />}
                        </button>
                    </div>
                </div>
            </nav >

            {/* Mobile Menu */}
            {
                isMobileMenuOpen && (
                    <div className="md:hidden fixed inset-0 z-40 bg-background/95 backdrop-blur-sm pt-20 px-6">
                        <div className="flex flex-col gap-4">
                            {navLinks.map((link) => (
                                <a
                                    key={link.name}
                                    href={link.href}
                                    className="text-lg font-medium text-foreground py-2 border-b border-border"
                                    onClick={() => setIsMobileMenuOpen(false)}
                                >
                                    {link.name}
                                </a>
                            ))}
                        </div>
                    </div>
                )
            }
        </>
    );
};

export default Navbar;
