---
layout: page
title: Find the Transform From Any Depth Sensor to Any Robot
img: assets/img/robo-icp/robo-icp.gif
description: My method finds the extrinsic transform from a depth sensor to a robot from a single view that includes rotationally unique robot features. This is useful for obtaining point clouds in the robot's base frame for visuomotor control, for egocentric and external depth sensor placements.
importance: 2
category: robotics
startDate: 2023-01-01
endDate: 2024-01-01
--- 

My method constructs a synthetic robot mesh using the robot’s URDF and joint positions, then applies [ICP](https://en.wikipedia.org/wiki/Iterative_closest_point) to align the mesh with the depth sensor’s view of the robot. This precisely estimates the sensor’s transform relative to a desired robot link. The approach generalizes across different robots and depth sensors, provided that the sensor captures a rotationally unique part of the robot, the depth cloud is cropped to primarily include robot points, and a rough initial sensor pose is given. For example, on a humanoid robot, a head-mounted depth camera can be localized relative to the head using a single view of the robot’s hands. Similarly, a depth camera mounted on a robot arm can be localized relative to the arm’s end by viewing the robot’s base links.

My code will hopefully be released soon!

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/robo-icp/robo-icp.gif" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Experimental matching results obtained with an externally mounted Intel Realsense L515 (on a tripod) and a Kinova Jaco 2, with Open3D doing most of the work while running inside on a container with ROS 2 Humble.
</div>



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/robo-icp/kinova.png" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    For some robots with shiny finishes, gaffer's tape can be applied to decrease specular reflection. This works well even with partial coverage!
</div>