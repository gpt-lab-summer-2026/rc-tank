# Autonomous AI Rover

An autonomous mobile rover combining a **locally hosted AI system** with embedded motor control. The project explores how a Raspberry Pi 5 and ESP32 can be used together to create a self-contained robotic platform capable of navigating its environment and avoiding obstacles.

## Overview

The original goal was to develop an autonomous rover that could roam freely, avoid obstacles, and report detected objects or events. The project follows a hardware architecture used by the research team for locally hosted AI and robotics systems:

* **Raspberry Pi 5** – AI models, image processing, and high-level rover control
* **ESP32** – Motor control and low-level firmware
* **Raspberry Pi Camera Module** – Live visual input
* **DC motors** – Rover locomotion
* **4-channel relay bridge** – Bidirectional motor control
* **USB serial** – Communication between the Raspberry Pi and ESP32

An existing RC rover was used as the mechanical base, allowing the project to focus primarily on the software, AI, and autonomous navigation aspects.

## System Architecture

The Raspberry Pi processes the camera feed and determines the appropriate direction of movement. Commands are sent to the ESP32 over USB serial, where the firmware controls the rover's motors through the relay bridge.

```text
┌─────────────────────┐
│   Raspberry Pi 5    │
│                     │
│  Camera Processing  │
│  AI / Object Detect │
│  Navigation Logic   │
└──────────┬──────────┘
           │
       USB Serial
           │
┌──────────▼──────────┐
│        ESP32        │
│                     │
│  Motor Control      │
│  Rover Firmware     │
└──────────┬──────────┘
           │
      GPIO Signals
           │
┌──────────▼──────────┐
│    Relay Bridge     │
└──────────┬──────────┘
           │
      DC Motors
           │
        Rover
```

The Raspberry Pi and motor system use separate power supplies. This prevents the high stall current of the DC motors from causing voltage drops that could reset or crash the computing hardware.

## Autonomous Navigation

The navigation software was implemented in Python using computer vision and image recognition techniques. The camera feed is analyzed to determine whether the path ahead is clear and which direction the rover should move.

The detection pipeline was developed to handle both:

* **Free path detection**
* **Obstacle detection and classification**

An object detection model was later incorporated to improve the rover's ability to distinguish obstacles from the surrounding environment.

The detection system was further tuned using additional training data collected from individual rooms. This room-specific data helped address differences in lighting, flooring, furniture, and other environmental conditions.

## Development Challenges

### Track Stiffness

The original RC rover tracks were not sufficiently rigid to support the additional Raspberry Pi, ESP32, battery, and electronics. The increased load caused:

* Excessive track friction
* Inconsistent movement
* Poor directional control
* Increased motor load and stalling

The original tracks were replaced with stiffer models. The control software was also adjusted to compensate for remaining mechanical ambiguity.

A wheeled trailer was additionally attached to the rover to support the majority of the system's weight. This introduced further challenges related to pulling resistance and the trailer shaft, which were addressed during development.

### Relay Bridge Activation

The relay control initially worked correctly when tested remotely through SSH. However, the rover behaved differently when the complete system was operating.

The cause was traced to the relay module sharing the motor power supply. Residual current in the relay circuit could continue powering the motors even after the relay state had changed.

The solution was to separate the power domains:

```text
Power Supply 1
└── Raspberry Pi
    └── ESP32
        └── Relay Module

Power Supply 2
└── DC Motors
```

This eliminated the unwanted motor activation and made the motor behavior consistent with the relay states.

## Camera & Detection Challenges

The camera feed became the primary source of software issues. The vision system initially struggled to distinguish the floor from obstacles.

For example:

* Table legs could sometimes be classified as part of the free path.
* Parts of the floor could be incorrectly classified as obstacles.
* Changes between rooms affected detection reliability.

The vision system was improved through a combination of:

* Additional training data
* Room-specific training data
* Obstacle detection
* Object detection
* Tunable detection parameters
* Iterative testing in different environments

These changes significantly improved the rover's ability to identify a navigable path.

## Technologies

**Hardware**

* Raspberry Pi 5
* ESP32
* Raspberry Pi Camera Module
* DC motors
* 4-channel relay module
* RC rover platform

**Software**

* Python
* Computer vision / image recognition OPENCV
* Object detection YOLO11
* ESP32 firmware
* USB serial communication

## Project Outcome

The project resulted in a functional prototype demonstrating autonomous rover navigation using a **locally hosted AI/vision system** and a dedicated embedded motor controller.

The development highlighted the importance of treating the mechanical, electrical, and software systems as an integrated platform. In particular, reliable autonomous navigation required not only improvements to the AI models but also mechanical modifications, isolated power supplies, reliable motor control, and environment-specific vision training.

(This readme was summarized from the project documentation usign AI)
