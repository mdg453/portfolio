
import React from 'react';
import { ExternalLink } from "lucide-react";
import { projects } from '../data/projects';
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "./ui/card" // We'll create a simple Card component manually since we skipped shadcn CLI

const Projects = () => {
    return (
        <section id="projects" className="py-24 bg-muted/30">
            <div className="container mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold text-foreground mb-4">Featured Projects</h2>
                    <div className="w-16 h-1 bg-primary rounded-full mx-auto mb-4"></div>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        Check out some of my recent work.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {projects.map((project, index) => (
                        <a
                            key={index}
                            href={project.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group block h-full no-underline"
                        >
                            <div className="bg-card rounded-xl border border-border shadow-sm overflow-hidden h-full flex flex-col transition-all hover:shadow-lg hover:-translate-y-1">
                                <div className="h-48 overflow-hidden relative">
                                    <img
                                        src={project.image}
                                        alt={project.title}
                                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                                    />
                                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                        <span className="text-white font-medium flex items-center gap-2">
                                            View Project <ExternalLink className="w-4 h-4" />
                                        </span>
                                    </div>
                                </div>

                                <div className="p-6 flex flex-col flex-grow">
                                    <h3 className="text-xl font-bold mb-2 group-hover:text-primary transition-colors">
                                        {project.title}
                                    </h3>
                                    <p className="text-muted-foreground mb-4 text-sm line-clamp-3">
                                        {project.description}
                                    </p>

                                    <div className="mt-auto flex flex-wrap gap-2">
                                        {project.tags.map((tag, i) => (
                                            <span key={i} className="px-2.5 py-0.5 rounded-full bg-secondary text-secondary-foreground text-xs font-medium">
                                                {tag}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </a>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Projects;
