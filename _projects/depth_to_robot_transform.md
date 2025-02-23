---
layout: page
title: Find the Transform From Any Depth Sensor to Any Robot
img: assets/img/robo-icp.gif
description: My method finds the extrinsic transform from a depth sensor to a robot from a single view that includes rotationally unique robot features. This is useful for obtaining point clouds in the robot's base frame, for egocentric and external depth sensor placements.
importance: 2
category: robotics
--- 

The method constructs a synthetic robot mesh from the robot's URDF/joint positions and uses ICP to find the transform from the robot mesh to the depth sensor's point cloud view of the robot, thus localizing the sensor relative to the robot. This works well across different robot models and depth sensors. My code will hopefully be released soon!

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/robo-icp.gif" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Experimental matching results obtained with an Intel Realsense L515 and a Kinova Jaco 2, with Open3D doing most of the work while running inside on a container with ROS 2 Humble.
</div>



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/kinova.png" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    For some robots with shiny finishes, gaffer's tape can be applied to decrease specular reflection. This works well even with partial coverage!
</div>