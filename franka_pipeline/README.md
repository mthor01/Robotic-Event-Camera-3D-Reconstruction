# franka_pipeline

This package implements a unified setup/pipeline to easily achieve multiple robotic tasks with the Franka Emika Panda Robot, to facilitate fast development for robotic applications. 

# Design Goals:
- Seamless switching sim / real
- Seamless switching remote control / policy
- Handle complexity through compartmentalization
	-> simple to reason about main loop, where all the setup is done
	-> sensible abstract base classes. If the interfaces are met, anything can be integrated
- (seamless switching between robot embodiments -> Not sure, how independent this actually is, since there are many cross dependencies)


# Design:

Components:
- sim / real switch (Deoxys/Robosuite)
- Agent Switch
    - provided
        - Robot state
        - Prompt
        - Image
    - VLA Agents via remote inference (OpenVLI, Pi0)
    - Human Control
        - Space Mouse
        - PS4 Controller
        - Gesture Control
- Sensor Switch
    - Cameras:
        - Realsense
        - Kinect

- Robot Controller Switch
    - Cartesian / Joint
    - ATTENTION: this could become more problematic

- Data Collection (seamless, into RLDS / OpenX-Robotics)
- Viewing data


# Design Decisions:
- remote inference is required for VLA agents, due to dependency conflicts -> make this first-class citizen




# Setup and Tutorials 
This whole framework is containerized for repeatable builds. To build the necessary containers, run the following commands:

    git clone git@github.com:David0tt/RobotReplicationDockerfiles.git
    cd RobotReplicationDockerfiles
    docker build -t deoxys_autostart docker_deoxys/
    DOCKER_BUILDKIT=1 docker build --ssh default -t openvla ./docker_openvla

Building the containers should take approximately ~1h. To build the containers, make sure ssh keys with the appropriate access to the github repositories used in the build are available in the environment. 


## To Launch

Follow the instructions from https://github.com/David0tt/RobotReplicationDockerfiles/blob/main/README_PANDA.md to start up the robot, unlock the joints and enable the FCI. Then run the deoxys docker container using 

Run the deoxys_autostart container. This one manages the backend of deoxys, i.e. the low-level C++ controller that sends commands directly to the robot arm via FCI. 

    docker run -it --rm -v /cshome:/cshome -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged deoxys_autostart

If everything works correctly, a tmux terminal with 3 panes should open, where on the right two terminals green info messages from the deoxys auto-scripts should be shown.


Now run the container for this packages frontend (for legacy reasons this is still called openvla [TODO])

    docker run -it --rm --runtime=nvidia --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged openvla

(If you plan to do continued development in this container, run without the `--rm` flag)

Open a terminal in this container (either using the existing one, vscode remote extension or starting a new terminal with `docker exec -it [CONTAINER_ID] bin/bash`). In it, you should now be able to run

    cd /workspace/franka_pipeline
    python main.py --real-robot

This should allow remote control of the robot via spacemouse. 

Alternatively try: 

    python main.py --osc-demo --real-robot

which should move the robot to four specified poses. 



# General Use of this package

The main entrypoint is the `main.py` which orchestrates all the correct sub-components to use from the CLI arguments. In general, run `python main.py --help` to get a list of all available options. 

Most notable are `--simulated-robot` / `--real-robot` to run either in simulation or on the real robot. I highly recommend testing any new code first in simulation. 

Most agents are wrapped using the EpisodeControlWrapperAgent. This allows keyboard control:
`q`: quit -> gracefully shut down the pipeline
`e`: end the episode -> saves the recorded episode data as a successful episode
`r`: reset the episode

# Usage Tutorials

## Run the Camera-Eye-in-Hand Calibration
### TODO
One finished calibration result for my current setup is already contained in this git, so you might don't need to run calibration if you are working on exactly the same robot. 

Place the calibration board in the front-center of the robot. Make sure the room is well lit and the lights are turned on. Then run

    python main.py --calibrate

The robot will move different prespecified calibration poses. Afterwards the detected calibration targets will be shown in an image viewer, and calibration will be done. 


## Run the MolMo-AnyGrasp Agent
To run the MolMo-AnyGrasp-Pipeline, you need to do the following:
1. Start the MolMo ZeroMQ server, which provides the detection of semantic objects in the image
2. Start the AnyGrasp ZeroMQ server, which provides the grasp point detection
3. Run the pipeline

The MolMo ZeroMQ server is run on the PC controlling the robot, however, in principle it could also run remotely (e.g. on a cluster). The AnyGrasp Server runs on on a different PC where AnyGrasp runs. To communicate with the server(s) on different PCs, port forwarding is used.

The specific startup is as follows: 

On the robot controlling PC, start two terminals. In one terminal, start the Molmo-ZMQ-Server. Copy the self-contained `molmo_zmq_server.py` file to this PC. First install the molmo conda environment following the instructions at the top of the `molmo_zmq_server.py`. Then run the server: 

    conda activate molmo
    cd ~/Nextcloud/MISC/molmo_zmq_server/
    python molmo_zmq_server.py
    # OR: CUDA_VISIBLE_DEVICES=$(python get_available_gpus.py) python molmo_zmq_server.py

In the second terminal, start the AnyGrasp-ZMQ-Server on the other PC (with port-forwarding):

    ssh -L 5562:127.0.0.1:5562 ott@panda1gpu
    cd ~/Nextcloud/AnyGrasp/anygrasp_sdk_modified/grasp_detection
    conda activate anygrasp
    python GraspDetectionClient.py


Then finally, in the docker environment with the franka_pipeline, run the pipeline: 

    # In simulation: 
    python main.py --molmo-grasp --object-query "Point to the red block"

    # On real robot (make sure a duplo block is in the scene):
    python main.py --real-robot --molmo-grasp --object-query "Point to the duplo block"

In any case, press `g` to trigger grasp detection and visiualization, and if you are happy with the grasp, press `h` to execute. 

# -> press g to trigger grasp detection and visualization

### TODO: You could also run the molmo server on a remote PC: 

    ssh -L 5583:127.0.0.1:5583 ott@avalon1   # for molmo


### Run OscPoseTargetDemoAgent
TODO

    python main.py --osc-demo --real-robot

### Run Benchmarking for best OSC-Pose-Target style control
This agent can be used to find optimal parameters for the initialization of the OscPoseTargetController. It employs a grid search over the specified benchmarking parameters

    python main.py --benchmark-osc-pose-target


### Run OscPoseTargetTestAgent
This Agent tries to approach 4 target poses from the 8 directions equally spaced on a circle around it on the plane specified by `--test-osc-pose-target-mode` (`xy` or `yz`). It produces a plot showing the start poses, desired target pose and actually reached poses. This can be useful to estimate the precision of OSC-Pose-Target style control.

    python main.py --test-osc-pose-target --test-osc-pose-target-mode xy --real-robot



## OpenVLA Server:
# TODO 



## Diverse Testing
### OscPoseTargetDemoAgent TODO


# Notes: 
# Notes: 
- The shared global state of the simulation is encapsulated in a SimEnv Class, which is given to the SimulatedRobotController and Simulated Camera
- For the real world this shared global state exists implicitly (it is the current state of the world) 
- make sure to run the realsense cameras with appropriate USB3.2 cables at appropriate USB3.2 connections, otherwise they might not correctly produce frames.

- Quirk: 
    - currently, before restarting you should also manually close the rerun viewer, as there sometimes occurs lagging with the live visualization otherwise
    - currently rerun just accumulates data over the whole time, which leads to heave memory usage. 

### A Note on "Agents"
Note, that in general in robotics there are two kinds of "agents". `Reactive agents` (e.g. `policy agents` in the RL sense), that expose an act function that can be querried for any timestep and returns relatively fast, and `Procedural agents` or `planning agents`, which do different actions or computations in a procedural fashion (e.g. calculate world model, plan action, move to position A, move to position B, ...). 

"Traditional robotics" mostly uses `procedural agents`, while "ML robotics" like RL or VLAs uses `reactive agents`. A low-level controller like `deoxys` generally lends itself well to the `reactive agents`. 

This is also the concept of the Agent class in this project. If you want to do a procedural agent, the best design pattern to do this is an explicit state machine, an example of this can be seen e.g. in the `MolMoAnyGraspAgent`.

```
class ProceduralAgent:
    def reset(self):
        self.state = "action1"

    def step(self, context):
        if self.state == "action1":
            done = self._do_action1(context)
            if done:
                self.state = "action2"

        elif self.state == "action2":
            done = self._do_action2(context)
            if done:
                self.state = "calc"

        elif self.state == "calc":
            self._do_some_calculation(context)
            self.state = "action3"

        elif self.state == "action3":
            done = self._do_action3(context)
            if done:
                self.state = "done"
```

### A Note on OSC_POSE_TARGET-Style control:
In many cases we want to move the robot to a specific location. Naively, this can be done with the OSC_POSE-type controller, which takes pose deltas, by simply calculating the difference between the current location and the target location as the action, and sending this action to the OSC_POSE controller. However, the OSC_POSE controller is a cartesian Impedance controller, that means it does not execute the action directly, but rather implements it as a desired state in a spring-damper system with some stiffness. This is generally a desired property, since it 
1. provides safe interaction, since the robot will not mindlessly apply arbitrary force to reach a location, and instead will stop at an obstructing object (e.g. a human)
2. provides better interaction with non-rigid objects
3. Allows smoother motion, reducing motor and robot wear

A disadvantage of the OSC_POSE cartesian impedance controller is however, that for very small actions, the action might not overcome the robots stiction, in particular in most cases the friction at the joints. Therefore, very small actions will not be executed well. For this reason, the naive approach of just computing the difference between the current location and the target location will stop at a short distance to the actually desired target where the robot stiction can not be overcome anymore. This prevents precise positioning at target locations. 

To mitigate this, multiple things could be done. First, we will describe the approach used here, which is adding an integral term of the current action, that slowly ramps up the action signal. Afterwards, some other alternatives and their drawbacks are discussed
We solve this problem by adding an integral gain term, when the current pose is close to the target pose. This integral gain slowly ramps up the action, so that small movement to the true desired target pose is facilitated. All this is implemented in the franka_pipeline.robot_controllers.OscPoseTargetController. The interface to use this from an agent is roughly like this:

    # Initialize the OscPoseTargetController with the target pose
    controller = OscPoseTargetController(target_pose)

    # In some loop: 
    action, is_finished = controller.calculate_action(current_pose)
    if is_finished:
        # Do change the state to whatever should be done after reaching the target. 

The OscPoseTargetController already uses sensible defaults for the different parameters of this approach, however, they can be modified at initialization. An example of a whole agent using OSC Pose Target style control can be seen in `osc_pose_target_demo_agent.py`.


Alternative Solutions to overcoming the real-world stiction for very small movements with the OSC_POSE impedance controller:
1. The stiffness Kp of the controller could be increased, however, this would result in less safe and more choppy movement
2. The action could just be scaled arbitrarily, however, this would result in more choppy movement
3. The OSC_POSE impedance controller could compensate for the individual joint stiction and friction. This would in principle be the "most correct" solution. However, this is generally a non-trivial control problem, since joint frictions are non-trivial at each individual joint. Further, the OSC_POSE controller "thinks" in cartesian impedance. However, the true cartesian impedance is non-linear and non-trivial, since it depends on the specific joint orientations and positions in the current state. So to correctly estimate and compensate the stiction/friction, first extensive movement testing at different orientations and speeds (with maximal freedom of movement and maximal speeds) would need to be done, to then be able to correctly compensate them. So in practice this proofs impractically difficult for our use case. 


# Good To Know

## On real world cameras, bandwidth and USB ports
You need to make sure that the connections of the cameras support the required speeds. This becomes problematic, in particular if multiple cameras are used at reasonably high resolutions and refresh rates. 

As a general rule of thumb, USB 3 should be used (in best case USB 3.2 (gen 2x2)), appropriate cables should be used, and multiple devices should be on different USB buses. 


in general you can examine the usb ports with 

    lsusb -t 

in general, `480M`means this is a USB 2 connection and `5000M` means this is a usb 3 connection. 

Further, you can examine which device is on which bus using 

    lsusb


You can get a list of available realsense camera configurations using 

    rs-enumerate-devices

The highest possible resultion for depth and image at the same time on a Intel RealSense D435 is 1280x720@30fps. 



## Procedural Agents vs. Reactive Agents
Note that there are generally two kinds of agents in robotics: procedural (also called structured or state-based) agents and reactive (also called policy-) agents. The procedural agents generally execute a number of steps, possibly gated by some conditions (e.g. detect the image, move to point A, grab, move to point B, ...). They are mostly how people are thinking about robotics from classical robotics (e.g. Sense-Plan-Act Framework, traditional motion planners). On the other hand, in ML / RL based robotics, we often think about an agent as executing some policy, that can be querried at each time step to produce some low-level action commands. This is a fundamental disconnect, where agents of these two types are generally not directly compatible. This framework naturally lends itself to reactive / policy-agents. 
If you want to integrate a procedural agent, a good design pattern to do this would be to use an explicit state machine. 

So for example, if you want to implement this naive procedural loop (in pseudocode): 

```
ProceduralAgent:
    def do_full_procedural_loop():
        self._do_action1()
        self._do_action2()
        self._do_some_calculation()
        self._do_action3()
        
    def _do_action1()
    def _do_action2()
    def _do_action3()
    def _do_some_calculation()
```

You should turn this into an explicit state machine:

```
    class ProceduralAgent:
        def reset(self):
            self.state = "action1"

        def step(self, context):
            if self.state == "action1":
                done = self._do_action1(context)
                if done:
                    self.state = "action2"

            elif self.state == "action2":
                done = self._do_action2(context)
                if done:
                    self.state = "calc"

            elif self.state == "calc":
                self._do_some_calculation(context)
                self.state = "action3"

            elif self.state == "action3":
                done = self._do_action3(context)
                if done:
                    self.state = "done"
```

An example of this implemented can be seen, for example in the `camera_eye_in_hand_calibration_agent.py` or `MolMoAnyGraspAgent.py`. 


## Logging

For clean logging, we use the standard `logging` framework. This is compatible with robosuite and deoxys. 
In general, at the top of any file import and instantiate a logger. 

    from franka_pipeline.logging import get_logger
    logger = get_logger(__name__)

Then anywhere you want to log, just use

    logger.info("My Info Message")
    logger.debug("My Debug Message")
    logger.warning("My Warning Message")
    logger.error("My Error Message")


The log-level can be set with cli arguments `--log-level` or `--verbose` / `-v`:

    python main.py --log-level DEBUG



# On getting Information from the agents to other parts of the code
Often, one needs to get more information than just the action from the agent to other parts of the code (e.g. for visualizing the planned grasp points, one would like to get these grasp points from a MolMoAnyGraspAgent to the rerun live visualization). Currently this is done by adding this information to the `metadata` returned by `agent.act()`. 
I am not entirely sure, whether this is the best interface to do this, but currently it works well.

# For live visualization do not forget to obtain the meshes from franka_description:
cd franka_pipeline/visualization
git clone https://github.com/frankarobotics/franka_description


# Steps that were taken to generate the urdfs:
On a machine with docker:

    git clone https://github.com/frankarobotics/franka_description
    cd franka_description
    ./scripts/create_urdf.sh fer --robot-ee franka_hand
    # copy the files created in urdf/ into franka_pipeline/visualization/urdfs


# TODO
- Implement everything that is missing
- in future maybe add setup.py


# Manual Installation instructions (without Docker):
[TODO] for now, just follow along roughly the instructions in the Dockerfile
# TODO put highest python version that is possible with deoxys client
conda create -n franka_pipeline python=3.10
pip install typer numpy
