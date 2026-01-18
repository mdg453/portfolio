// --- Piano Tuner Project: 3-String Mechanism ---
// Parametric Design for OpenSCAD
// Units: mm

// --- CONTROL FLAGS ---
// Set this to true ONLY when you want to export the DXF file for the workshop.
// Set to false to view the 3D assembly.
generate_dxf = false; 

// Resolution optimization
// High res for DXF export, Low res for 3D preview to prevent freezing
$fn = generate_dxf ? 60 : 30; 

// --- Parameters ---
wood_thickness = 25;
base_width = 180;
base_length = 650;

// Hardware Dimensions
nema17_hole_dist = 31;
tuner_hole_dia = 10;   
rod_diameter = 12;     
rod_spacing = 150;     

// Electronics Dimensions
pickup_width = 70;     
pickup_depth = 18;     
solenoid_plunger_d = 8; 
solenoid_mount_dist = 20; 

// String Geometry
scale_length = 400;    
bridge_pos = 80;       
nut_pos = bridge_pos + scale_length;
bridge_height = 15;    

// Nut/Saddle Dimensions
nut_thickness = 5;     
nut_height = 8;        
nut_slot_depth = 4;    

// Spacing
string_spacing_active = 20; 
tuner_spacing_head = 55;    

// Feet Dimensions
feet_height = 15;      
feet_diameter = 25;

// --- Modules (Cuts & Mounts) ---

module nema17_mount() {
    cylinder(h=wood_thickness*3, d=24, center=true); 
    for(x=[-1,1]) for(y=[-1,1]) {
        translate([x*nema17_hole_dist/2, y*nema17_hole_dist/2, 0])
        cylinder(h=wood_thickness*3, d=3.5, center=true);
    }
}

module pickup_cutout() {
    cube([pickup_width, pickup_depth, wood_thickness+2], center=true);
    for(x=[-1,1]) 
        translate([x*(pickup_width/2 + 4), 0, 0])
        cylinder(h=wood_thickness*3, d=3, center=true);
}

module solenoid_mount() {
    cylinder(h=wood_thickness*3, d=solenoid_plunger_d, center=true);
    for(y=[-1,1])
        translate([0, y*solenoid_mount_dist/2, 0])
        cylinder(h=wood_thickness*3, d=3, center=true);
}

module tuner_hole() {
    cylinder(h=wood_thickness*3, d=tuner_hole_dia, center=true);
}

module foot() {
    translate([0, 0, -wood_thickness/2 - feet_height/2])
        cylinder(h=feet_height, d=feet_diameter, center=true);
}

// --- Visual Components (Hardware Models) ---

module vis_nema17() {
    color("DarkSlateGray") {
        difference() {
            cube([42.3, 42.3, 34], center=true); 
            for(x=[-1,1]) for(y=[-1,1]) 
                translate([x*15.5, y*15.5, 17]) cylinder(h=2, d=3, center=true);
        }
    }
    color("Silver") {
        translate([0,0,17]) cylinder(h=24, d=5, center=false); 
        translate([0,0,17]) cylinder(h=2, d=22, center=false); 
    }
}

module vis_pickup() {
    color("Ivory") {
        translate([0,0,2]) cube([pickup_width-4, pickup_depth-4, 12], center=true); 
        cube([pickup_width+15, pickup_depth+5, 2], center=true); 
    }
    color("Silver") 
        for(i=[-2.5:1:2.5]) translate([i*10, 0, 8]) cylinder(h=2, d=5, center=true);
}

module vis_solenoid() {
    color("Gold") { 
        rotate([90,0,0]) cylinder(h=30, d=20, center=true); 
        translate([0,0,15]) cube([25, 5, 2], center=true); 
    }
    color("Silver") { 
        cylinder(h=40, d=4, center=true);
        translate([0,0,20]) sphere(d=6); 
    }
}

module vis_tuner_peg() {
    color("Silver") {
        cylinder(h=35, d=6, center=true); 
        translate([0,0,10]) cylinder(h=2, d=14, center=true); 
    }
    color("LightGrey") { 
        translate([0, -15, -10]) cube([15, 20, 10], center=true); 
        translate([0, -25, -10]) rotate([0,90,0]) cylinder(h=20, d=8, center=true); 
    }
}

module vis_rod() {
    color("Silver") 
    rotate([90, 0, 0]) 
    cylinder(h=base_length, d=rod_diameter, center=true);
}

module vis_screw() {
    color("Black") cylinder(h=4, d=5, $fn=6);
}

module vis_nut_insert() {
    color("Ivory") 
    cube([base_width, nut_thickness, nut_height], center=true);
}

module label_text(txt) {
    linear_extrude(height = 1) {
        text(txt, size=15, halign="center");
    }
}

module label_text_small(txt) {
    linear_extrude(height = 1) {
        text(txt, size=8);
    }
}

// --- Module: The Wooden Board (Manufacturing Part) ---
module wooden_board() {
    difference() {
        // Base Plate & Blocks
        union() {
            color("BurlyWood") cube([base_width, base_length, wood_thickness], center=true);
            
            // Bridge Block
            translate([0, -base_length/2 + bridge_pos, wood_thickness/2 + bridge_height/2])
                color("Sienna") cube([base_width, 25, bridge_height], center=true);

            // Nut Block
            translate([0, -base_length/2 + nut_pos, wood_thickness/2 + bridge_height/2])
                color("Sienna") cube([base_width, 25, bridge_height], center=true);
        }

        // Subtractions
        // A. Rod Channels
        for (side = [-1, 1]) {
            translate([side * rod_spacing / 2, 0, 0]) 
            rotate([90, 0, 0]) 
            cylinder(h=base_length+10, d=rod_diameter, center=true);
        }
        
        // B. NUT & SADDLE SLOTS
        translate([0, -base_length/2 + bridge_pos, wood_thickness/2 + bridge_height - nut_slot_depth/2 + 0.1])
             cube([base_width+2, nut_thickness, nut_slot_depth], center=true);

        translate([0, -base_length/2 + nut_pos, wood_thickness/2 + bridge_height - nut_slot_depth/2 + 0.1])
             cube([base_width+2, nut_thickness, nut_slot_depth], center=true);

        // C. Per-String Holes
        for (i = [-1, 0, 1]) {
            x_string = i * string_spacing_active;
            x_tuner = i * tuner_spacing_head;
            y_tuner = base_length/2 - 60 - (abs(i)*35);
            
            // Anchor Holes
            translate([x_string, -base_length/2 + bridge_pos - 35, 0])
                cylinder(h=wood_thickness*3, d=4, center=true);
            // Grooves
            translate([x_string, -base_length/2 + bridge_pos, wood_thickness/2 + bridge_height/2])
                rotate([90, 0, 0]) cylinder(h=40, d=3, center=true);
            translate([x_string, -base_length/2 + nut_pos, wood_thickness/2 + bridge_height/2])
                rotate([90, 0, 0]) cylinder(h=40, d=3, center=true);
            // Tuners & Motors
            translate([x_tuner, y_tuner, 0]) tuner_hole();
            translate([x_tuner, y_tuner + 45, 0]) nema17_mount();
            // Solenoid
            translate([x_string, -base_length/2 + bridge_pos + scale_length*0.12, 0])
                 solenoid_mount();
            // Pickup
            translate([x_string, -base_length/2 + bridge_pos + scale_length*0.75, 0])
                pickup_cutout();
        }
    }
}

// --- Module: Hardware Assembly (Components Only) ---
module hardware_assembly() {
    for(x=[-1,1]) for(y=[-1,1]) {
        translate([x*(base_width/2 - 20), y*(base_length/2 - 20), 0])
        color("Black") foot();
    }
    
    for (side = [-1, 1]) {
        translate([side * rod_spacing / 2, 0, 0]) vis_rod();
    }
    
    translate([0, -base_length/2 + bridge_pos, wood_thickness/2 + bridge_height - nut_slot_depth + nut_height/2])
        vis_nut_insert();
    translate([0, -base_length/2 + nut_pos, wood_thickness/2 + bridge_height - nut_slot_depth + nut_height/2])
        vis_nut_insert();

    for (i = [-1, 0, 1]) {
        x_string = i * string_spacing_active;
        x_tuner = i * tuner_spacing_head;
        y_tuner = base_length/2 - 60 - (abs(i)*35);
        
        translate([x_tuner, y_tuner + 45, wood_thickness/2 + 17]) vis_nema17();
        translate([x_tuner, y_tuner, wood_thickness/2]) vis_tuner_peg();
        translate([x_string, -base_length/2 + bridge_pos + scale_length*0.75, wood_thickness/2 - 5]) vis_pickup();
        translate([x_string, -base_length/2 + bridge_pos + scale_length*0.12, -wood_thickness/2 - 10]) vis_solenoid();
        translate([x_string, -base_length/2 + bridge_pos + scale_length*0.12 - 10, wood_thickness/2]) vis_screw();
        translate([x_string, -base_length/2 + bridge_pos + scale_length*0.12 + 10, wood_thickness/2]) vis_screw();
    }
    
    // Low-poly string visualization for performance
    color("gold")
    for (i = [-1, 0, 1]) {
        x_string = i * string_spacing_active;
        x_tuner = i * tuner_spacing_head;
        y_tuner = base_length/2 - 60 - (abs(i)*35);
        z_string = wood_thickness/2 + bridge_height + (nut_height - nut_slot_depth) + 1;
        
        hull() {
            translate([x_string, -base_length/2 + bridge_pos - 35, 0]) sphere(d=1.5, $fn=12);
            translate([x_string, -base_length/2 + bridge_pos, z_string]) sphere(d=1.5, $fn=12);
        }
        hull() {
            translate([x_string, -base_length/2 + bridge_pos, z_string]) sphere(d=1.5, $fn=12);
            translate([x_string, -base_length/2 + nut_pos, z_string]) sphere(d=1.5, $fn=12);
        }
        hull() {
            translate([x_string, -base_length/2 + nut_pos, z_string]) sphere(d=1.5, $fn=12);
            translate([x_tuner, y_tuner, wood_thickness/2 + 5]) sphere(d=1.5, $fn=12);
        }
    }
}

// --- Module: BOM ---
module parts_legend() {
    translate([0, 200, 0]) {
        vis_nema17();
        color("Black") translate([40, 0, 0]) label_text_small("NEMA 17 Motor");
    }
    translate([0, 120, 0]) {
        rotate([0,0,90]) vis_tuner_peg();
        color("Black") translate([40, 0, 0]) label_text_small("Guitar Tuner");
    }
    translate([0, 50, 0]) {
        vis_pickup();
        color("Black") translate([40, 0, 0]) label_text_small("Single Coil Pickup");
    }
    translate([0, -20, 0]) {
        vis_solenoid();
        color("Black") translate([40, 0, 0]) label_text_small("Push Solenoid (12V)");
    }
    translate([0, -80, 0]) {
        rotate([0,0,90]) vis_nut_insert();
        color("Black") translate([20, 0, 0]) label_text_small("Bone Nut/Saddle");
    }
    translate([0, -140, 0]) {
        color("Black") foot();
        color("Black") translate([40, 0, 0]) label_text_small("Rubber Foot");
    }
    translate([0, -180, 0]) {
        vis_screw();
        color("Black") translate([40, 0, 0]) label_text_small("M3 Screw");
    }
    translate([-50, 0, 0]) {
        rotate([0, 90, 0]) vis_rod();
        color("Black") translate([-20, 0, 0]) rotate([0,0,90]) label_text_small("Steel/Alu Rod 12mm");
    }
}

// --- LOGIC SWITCH ---
if (generate_dxf) {
    // 2D Export Mode: Projects the board to 2D
    // cut=false creates a "shadow" of the whole object (good for seeing blocks)
    // cut=true slices at Z=0 (good for just the base plate holes)
    // We use projection() on the clean board module only.
    projection(cut = false) wooden_board();
} else {
    // 3D Visualization Mode
    translate([-base_width - 60, 0, 0]) {
        wooden_board();
        color("Black") translate([0, base_length/2 + 40, 0]) label_text("1. Production Part");
    }

    translate([base_width/2, 0, 0]) {
        wooden_board();
        hardware_assembly();
        color("Black") translate([0, base_length/2 + 40, 0]) label_text("2. Assembly Reference");
    }

    translate([base_width * 2 + 40, 0, 0]) {
        parts_legend();
        color("Black") translate([0, base_length/2 + 40, 0]) label_text("3. Hardware Kit (BOM)");
    }
}