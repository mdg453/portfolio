
import React from 'react';

const Skills = () => {
    return (
        <section id="skills" className="py-24 bg-background">
            <div className="container mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold text-foreground mb-4">Skills & Technologies</h2>
                    <div className="w-16 h-1 bg-primary rounded-full mx-auto mb-4"></div>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        The main technologies I use in my daily development workflow.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {[
                        {
                            title: "Languages",
                            skills: ['Python', 'C++', 'Java', 'Kotlin', 'SQL', 'Bash']
                        },
                        {
                            title: "Primary Focus",
                            skills: [
                                'Verification Automation',
                                'Embedded Systems',
                                'HW-SW Integration',
                                'Firmware Validation',
                                'Computer Architecture'
                            ]
                        },
                        {
                            title: "Tools & DevOps",
                            skills: ['Docker', 'Kubernetes', 'Jenkins', 'Linux', 'Splunk', 'Git']
                        }
                    ].map((category, idx) => (
                        <div key={idx} className="bg-muted/30 p-8 rounded-2xl border border-border/50 hover:border-primary/50 transition-colors">
                            <h3 className="text-xl font-semibold mb-6 text-primary text-center">{category.title}</h3>
                            <div className="flex flex-wrap gap-3 justify-center">
                                {category.skills.map(skill => (
                                    <span
                                        key={skill}
                                        className="px-3 py-1.5 bg-background border border-border rounded-lg text-sm text-foreground hover:border-primary hover:text-primary transition-all shadow-sm"
                                    >
                                        {skill}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Skills;
