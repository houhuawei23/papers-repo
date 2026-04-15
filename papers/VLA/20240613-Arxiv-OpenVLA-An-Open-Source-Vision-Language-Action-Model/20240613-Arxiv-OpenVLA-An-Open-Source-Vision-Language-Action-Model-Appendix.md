<a id="appendix-a"></a>

## Appendix A Data Mixture Details

We list our used data mixture in [Table 3](#table-3). The mixture mostly follows [[5]], with a few additional datasets.

<a id="table-3"></a>

> Table 3: OpenVLA training data mixture using datasets from the Open X-Embodiment dataset [[1]], following [[5]] with a few additions.

| OpenVLA Training Dataset Mixture    |                                                                                                                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fractal [[92]]                      | 12.7%                                                                                                                                                                                       |
| Fractal [[92]]                      | 12.7%                                                                                                                                                                                       |
| Kuka [[45]]                         | 12.7%                                                                                                                                                                                       |
| Kuka [[45]]                         | 12.7%                                                                                                                                                                                       |
| Bridge[[47], [6]]                   | 13.3%                                                                                                                                                                                       |
| Bridge[[47], [6]]                   | 13.3%                                                                                                                                                                                       |
| Taco Play [[93], [94]]              | 3.0%                                                                                                                                                                                        |
| Taco Play [[93], [94]]              | 3.0%                                                                                                                                                                                        |
| Jaco Play [[95]]                    | 0.4%                                                                                                                                                                                        |
| Jaco Play [[95]]                    | 0.4%                                                                                                                                                                                        |
| Berkeley Cable Routing [[96]]       | 0.2%                                                                                                                                                                                        |
| Berkeley Cable Routing [[96]]       | 0.2%                                                                                                                                                                                        |
| Roboturk [[97]]                     | 2.3%                                                                                                                                                                                        |
| Roboturk [[97]]                     | 2.3%                                                                                                                                                                                        |
| Viola [[98]]                        | 0.9%                                                                                                                                                                                        |
| Viola [[98]]                        | 0.9%                                                                                                                                                                                        |
| Berkeley Autolab UR5 [[99]]         | 1.2%                                                                                                                                                                                        |
| Berkeley Autolab UR5 [[99]]         | 1.2%                                                                                                                                                                                        |
| Toto [[100]]                        | 2.0%                                                                                                                                                                                        |
| Toto [[100]]                        | 2.0%                                                                                                                                                                                        |
| Language Table [[101]]              | 4.4%                                                                                                                                                                                        |
| Language Table [[101]]              | 4.4%                                                                                                                                                                                        |
| Stanford Hydra Dataset [[102]]      | 4.4%                                                                                                                                                                                        |
| Stanford Hydra Dataset [[102]]      | 4.4%                                                                                                                                                                                        |
| Austin Buds Dataset [[103]]         | 0.2%                                                                                                                                                                                        |
| Austin Buds Dataset [[103]]         | 0.2%                                                                                                                                                                                        |
| NYU Franka Play Dataset [[104]]     | 0.8%                                                                                                                                                                                        |
| NYU Franka Play Dataset [[104]]     | 0.8%                                                                                                                                                                                        |
| Furniture Bench Dataset [[105]]     | 2.4%                                                                                                                                                                                        |
| Furniture Bench Dataset [[105]]     | 2.4%                                                                                                                                                                                        |
| UCSD Kitchen Dataset [[106]]        | <0.1%                                                                                                                                                                                       |
| UCSD Kitchen Dataset [[106]]        | <0.1%                                                                                                                                                                                       |
| Austin Sailor Dataset [[107]]       | 2.2%                                                                                                                                                                                        |
| Austin Sailor Dataset [[107]]       | 2.2%                                                                                                                                                                                        |
| Austin Sirius Dataset [[108]]       | 1.7%                                                                                                                                                                                        |
| Austin Sirius Dataset [[108]]       | 1.7%                                                                                                                                                                                        |
| DLR EDAN Shared Control [[109]]     | <0.1%                                                                                                                                                                                       |
| DLR EDAN Shared Control [[109]]     | <0.1%                                                                                                                                                                                       |
| IAMLab CMU Pickup Insert [[110]]    | 0.9%                                                                                                                                                                                        |
| IAMLab CMU Pickup Insert [[110]]    | 0.9%                                                                                                                                                                                        |
| UTAustin Mutex [[111]]              | 2.2%                                                                                                                                                                                        |
| UTAustin Mutex [[111]]              | 2.2%                                                                                                                                                                                        |
| Berkeley Fanuc Manipulation [[112]] | 0.7%                                                                                                                                                                                        |
| Berkeley Fanuc Manipulation [[112]] | 0.7%                                                                                                                                                                                        |
| CMU Stretch [[113]]                 | 0.2%                                                                                                                                                                                        |
| CMU Stretch [[113]]                 | 0.2%                                                                                                                                                                                        |
| BC-Z [[55]]                         | 7.5%                                                                                                                                                                                        |
| BC-Z [[55]]                         | 7.5%                                                                                                                                                                                        |
| FMB Dataset [[114]]                 | 7.1%                                                                                                                                                                                        |
| FMB Dataset [[114]]                 | 7.1%                                                                                                                                                                                        |
| DobbE [[115]]                       | 1.4%                                                                                                                                                                                        |
| DobbE [[115]]                       | 1.4%                                                                                                                                                                                        |
| DROID [[11]]                        | 10.0%(^6^66We remove DROID for the last third of training due to slow learning progress (see [Section 3.3](#section-3-3)) and re-distribute its mixture weights across all other datasets.) |
| DROID [[11]]                        | 10.0%(^6^66We remove DROID for the last third of training due to slow learning progress (see [Section 3.3](#section-3-3)) and re-distribute its mixture weights across all other datasets.) |

<a id="appendix-b"></a>

## Appendix B Evaluation Tasks and Detailed Results

In this section, we provide more details on the BridgeData V2 WidowX and Google robot evaluations discussed in [Section 5.1](#section-5-1), as well as the Franka-Tabletop and Franka-DROID fine-tuning evaluations discussed in [Section 5.2](#section-5-2).

- [B.1 BridgeData V2 WidowX Evaluation Details](#b1-bridgedata-v2-widowx-evaluation-details)
- [B.2 Google Robot Evaluation Details](#b2-google-robot-evaluation-details)
- [B.3 Data-Efficient Adaptation Experiment Details](#b3-data-efficient-adaptation-experiment-details)

### B.1 BridgeData V2 WidowX Evaluation Details

Here we focus specifically on BridgeData V2 evaluations discussed in [Section 5.1](#section-5-1).

- [B.1.1 BridgeData V2 Evaluation Tasks](#b11-bridgedata-v2-evaluation-tasks)
- [B.1.2 Comparing Evaluation Tasks to Original BridgeData V2 Training Data](#b12-comparing-evaluation-tasks-to-original-bridgedata-v2-training-data)
- [B.1.3 Detailed BridgeData V2 Evaluation Results](#b13-detailed-bridgedata-v2-evaluation-results)

#### B.1.1 BridgeData V2 Evaluation Tasks

As described in [Section 5.1](#section-5-1), we evaluate each generalist robot manipulation policy on 17 tasks with 10 trials each. In this section, we provide details on the task categories and individual tasks.

<a id="figure-6"></a>

![openvla_teaser](images/openvla_teaser.png)

> Figure 6: BridgeData V2 WidowX robot evaluation tasks. We evaluate every generalist robot policy on 4 types out-of-distribution (OOD) generalization tasks: visual, motion, physical, and semantic (as defined in [Section 5.1](#section-5-1)). Every pair of images shows the start state and an example end state after the robot completes the task. We also rigorously assess language grounding in the 3 tasks shown in the bottom 3 rows, by changing the prompt while fixing the initial state and testing whether the policy can approach the correct target object.

In total, we evaluate on 5 visual generalization tasks, 2 motion generalization tasks, 3 physical generalization tasks, 4 semantic generalization tasks, and 3 language grounding tasks. Note that all tasks we evaluate on introduce some form of distribution shift since we are unable to procure the exact objects used in the original dataset (other distribution shifts naturally arise as we reproduce a real-world test environment originally constructed at a different location; see [Section B.1.2](https://arxiv.org/html/2406.09246v3#A2.SS1.SSS2) for a detailed discussion on such distribution shifts). All 17 tasks are depicted in [Fig. 6](https://arxiv.org/html/2406.09246v3#A2.F6). Each rollout is marked as a failure (0) or success (1). In some more difficult tasks, we record partial successes (0.5); we describe the conditions for partial credit in the task descriptions below.

Below we describe each of the 17 tasks, in the order shown in [Fig. 6](https://arxiv.org/html/2406.09246v3#A2.F6):

- 1. Put Eggplant into Pot (Easy Version): The robot’s goal is to pick up the eggplant and drop it into the pot. This is a visual generalization task because we use a handcrafted paper pot that has a different appearance than the pot used in the original BridgeData V2 training dataset (since we are unable to procure the original pot). Unlike all 16 other tasks, for this particular task we initialize the robot’s end-effector directly above the eggplant before rolling out the policy; hence, we call this the “Easy Version” of the “Put Eggplant into Pot” task.
- 2. Put Eggplant into Pot: This is the same task as described above, except that the robot’s end-effector is not initialized directly above the eggplant. Instead, we initialize it in a position that is fixed across all rollouts, which means that the robot must horizontally reach for the eggplant first before manipulating it. (Note: The same applies to all other tasks described below.) This is a visual generalization task for the same reason as above.
- 3. Put Cup from Counter into Sink: The robot’s goal is to pick up the pink cup from either the kitchen countertop or drying rack and place it into the sink on the right. This is a visual generalization task because we use a pink cup rather than a blue cup (a blue cup is used in the original BridgeData V2 dataset, but we find that none of the methods we evaluate is able to manipulate it reliably – most likely because the color of the cup blends in with the color of the sink).
- 4. Put Eggplant into Pot (w/ Clutter): This is the same task as the “Put Eggplant into Pot” task, except that it is more difficult due to the presence of several distractor objects. It is a visual generalization task for the same reason discussed in the normal “Put Eggplant into Pot” task, and even more so given unseen distractors in the scene. _Partial credit (0.5 out of 1) is rewarded when the robot moves towards the correct target object._
- 5. Put Yellow Corn on Pink Plate: The robot’s goal is to pick up the yellow corn and place it on the pink plate. This is a visual generalization task due to the presence of unseen distractor objects in the scene, such as a green dinosaur on the countertop in the back section of the sink. _Partial credit (0.5 out of 1) is rewarded when the robot moves towards the correct target object._
- 6. Lift Eggplant: The robot’s goal is to grasp and lift the eggplant into the air. This is a motion generalization task because the eggplant is initialized in unseen positions and/or orientations, and the robot is forced to move beyond its training distribution of positions and/or orientations and often perform long-range reaching in order to complete the task. (Note: Long-range reaching is not demonstrated in this environment in the original BridgeData V2 demonstrations; see [Section B.1.2](https://arxiv.org/html/2406.09246v3#A2.SS1.SSS2) for details.) We find that this task, though seemingly simple, is deceptively challenging for many policies. _Partial credit (0.5 out of 1) is rewarded when the robot makes contact with the eggplant._
- 7. Put Carrot on Plate (w/ Height Change): The robot’s goal is to pick up the carrot and place it on the yellow plate. This is a motion generalization task because the plate is elevated from its usual position at the bottom of the sink, and the robot must adjust its trajectory to correctly place the carrot on the elevated platform (without knocking down the plate in the process). _Partial credit (0.5 out of 1) is rewarded when the robot grasps the carrot and touches the plate with it._
- 8. Put Carrot on Plate: This is the same task as above, except that the plate is at its normal position (at the bottom of the sink or drying rack). We consider this a physical generalization task because the carrot has a different size and shape than the one used in the original BridgeData V2 dataset, which is shorter and narrower. (Note that the previous version of this task listed above would also technically be a physical generalization task since it involves the same carrot, but we list it under the “motion generalization” category since that is the focus there.)
- 9. Flip Pot Upright: The robot’s goal is to manipulate the pot such that it is oriented upright in the sink at the end of the episode. This is a physical generalization task because this pot has a different size and shape than the one used in the original BridgeData V2 training demonstrations (the pot we use is wider and shorter).
- 10. Lift AAA Battery: The robot’s goal is simply to grasp the AAA battery and lift it up into the air. This is considered a physical generalization task because the battery is much smaller and thinner than target objects seen in the BridgeData V2 training demonstrations in this environment; see [Section B.1.2](https://arxiv.org/html/2406.09246v3#A2.SS1.SSS2) for details. (Note that this target object does not exist in the original BridgeData V2 demonstrations in this environment, so this is also an instance of “semantic generalization”, but we classify it solely as “physical generalization” since that is the main focus here).
- 11. Move Skull into Drying Rack: The robot’s goal is to grasp the skull windup toy and drop it into the yellow drying rack in the left part of the sink. This is a semantic generalization task since the skull is an unseen target object (does not appear in the BridgeData V2 training demonstrations).
- 12. Lift White Tape: The robot’s goal is to grasp and lift the white roll of tape into the air. This is a semantic generalization task since the white tape roll is an unseen target object (does not appear in the BridgeData V2 training demonstrations). (Note that this task may also be considered as “physical generalization” because of its shape being different than the objects seen in the training demonstrations in this environment; most policies struggle to grasp objects with this ring structure, and they often move the robot’s end-effector directly into the center region.)
- 13. Take Purple Grapes out of Pot: The robot’s goal is to grasp the purple grapes lying inside the steel pot and remove it from the pot (by lifting it out and/or dropping it anywhere outside the pot). This is a semantic generalization task because it is an unseen language instruction; the robot has never seen this task in the original BridgeData V2 training dataset.
- 14. Stack Blue Cup on Pink Cup: The robot’s goal is to grasp the blue cup and place it securely on top of the pink cup. This is a semantic generalization task because it is an unseen language instruction; the robot has never seen this task in this environment in the original BridgeData V2 training dataset. _Partial credit (0.5 out of 1) is rewarded when the robot grasps the blue cup and touches the pink cup with the blue cup._
- 15. Put {Eggplant, Red Bottle} into Pot: This is a language grounding task. The robot’s goal is to put the specified target object into the pot. Both the eggplant and red bottle are present in the scene. We conduct paired evaluations: for the same initial state, we prompt the policy to target the eggplant in one episode, and then the red bottle in the next episode. We test each method 5 times with the eggplant and 5 times with the red bottle, using the same set of 5 initial states for both target objects. _Partial credit (0.5 out of 1) is rewarded when the robot moves towards the correct target object._
- 16. Lift {Cheese, Red Chili Pepper}: This is a language grounding task. The robot’s goal is to grasp and lift the specified target object. We conduct paired evaluations as described in the task above. _Partial credit (0.5 out of 1) is rewarded when the robot moves towards the correct target object._
- 17. Put {Blue Cup, Pink Cup} on Plate: This is a language grounding task. The robot’s goal is to grasp the specified target object and place it onto the plate. We conduct paired evaluations as described in other language grounding tasks. _Partial credit (0.5 out of 1) is rewarded when the robot moves towards the correct target object._

#### B.1.2 Comparing Evaluation Tasks to Original BridgeData V2 Training Data

We conduct our evaluations in a sink environment used in the original BridgeData V2 dataset [[6]]. We reproduce the environment to match the original environment in the BridgeData V2 dataset with rough approximations for the robot’s location relative to the sink, as well as the camera’s placement relative to the scene. Given the lack of precise measurements of these positions in the original dataset, we are unable to reproduce the _exact_ environment setup, and natural distribution shifts arise due to slightly different robot, sink, and camera placements. In addition, since we evaluate robot policies in a different location than where the training demonstrations were collected from, other natural distribution shifts arise. For example, the lighting conditions and background (e.g., visible areas behind the sink) are inevitably different than what was seen in the training dataset. Furthermore, we are unable to procure the exact set of objects used in the original BridgeData V2 dataset, so there are distribution shifts between the objects used at train time and those used at test time.

Despite all these challenges, we find that certain generalist policies, such as OpenVLA and RT-2-X, can still generalize and perform various tasks fairly reliably “out-of-the-box”. Other generalist policies, such as RT-1-X and Octo, can also complete some tasks, though they struggle when tested with more difficult generalization tasks in our BridgeData V2 evaluation suite.

The original BridgeData V2 dataset includes demonstrations of the following seven tasks in this specific sink environment: “Flip Pot Upright”, “Put Carrot on Plate”, “Put Cup from Counter (or Drying Rack) into Sink”, “Put Eggplant into Pot”, “Put Knife on Cutting Board”, “Put Spoon in Pot”, and “Turn Lever Vertical to Front”. See [Fig. 7](https://arxiv.org/html/2406.09246v3#A2.F7) for samples images of all these tasks from the original dataset. Note that all training demonstrations collected in this environment are initialized such that the robot’s end-effector is positioned directly above the target object in the beginning of the episode. (However, this is not the case across all environments in the BridgeData V2 dataset; in some other environments, the robot is initialized farther away from the target object, so it must horizontally reach for the object first before manipulating it.)

<a id="figure-7"></a>

![openvla_teaser](images/openvla_teaser.png)

> Figure 7: Original BridgeData V2 sink environment tasks. Images from sample demonstrations in the sink environment from the original BridgeData V2 dataset reveal that all demonstrations in this environment were initialized such that the robot’s end-effector was positioned immediately above the target object. Note that these initial states are different from the initial states we use in our BridgeData V2 evaluation tasks shown in [Fig. 6](https://arxiv.org/html/2406.09246v3#A2.F6). In our evaluations, we always initialize the robot’s end-effector to a fixed location above the sink, rather than positioning it directly above the target object (except for one task: “Put Eggplant into Pot (Easy Version)”).

In our BridgeData V2 evaluation suite, only one task – “Put Eggplant into Pot (Easy Version”) – is initialized with the robot’s end-effector hovering directly over the target object; in all 16 other tasks, the end-effector is initialized at a fixed location above the sink such that the robot must horizontally reach towards the object. This initial condition, in combination with the distribution shifts we introduce in the various types of OOD generalization in our evaluation suite, challenges the generalist policies and requires a high degree of robustness in order to complete the tasks successfully. Hence, the success rates for policies like RT-1-X and Octo are lower than what is reported in prior works. However, we find that other policies such as RT-2-X and OpenVLA still achieve relatively strong performance despite all these distribution shifts and challenges.

#### B.1.3 Detailed BridgeData V2 Evaluation Results

See [Table 4](#table-4) for the full BridgeData V2 WidowX evaluation results. The number of successes for each method, out of 10 trials, is listed for each of 17 tasks. OpenVLA achieves strongest performance in the majority of the tasks and has the highest aggregate success rate among the generalist policies. RT-2-X also shows good performance, outperforming RT-1-X and Octo, though it does not perform as well as OpenVLA. RT-1-X and Octo generally experience difficulty in these generalization tasks.

<a id="table-4"></a>

> Table 4: Detailed BridgeData V2 WidowX evaluation results. We report performance on the full evaluation suite of 17 tasks (discussed in [Section 5.1](#section-5-1)), including visual/motion/physical/semantic generalization tasks and language grounding tasks. Note that _partial success_ (score of 0.5) is possible for some tasks; see [Section B.1.1](https://arxiv.org/html/2406.09246v3#A2.SS1.SSS1) for details. We find that OpenVLA performs best in most tasks and achieves highest performance overall, followed by RT-2-X. On the other hand, RT-1-X and Octo struggle in the evaluations, only getting 0–2 successes in several tasks. See [Fig. 6](https://arxiv.org/html/2406.09246v3#A2.F6) for illustrations of all tasks.

| Category           | Task                                   | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| ------------------ | -------------------------------------- | ----------------- | ------------------ | ---------------- | ------------------ | -------------------------- |
| Category           | Task                                   | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category           | Task                                   | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category           | Task                                   | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category           | Task                                   | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category           | Task                                   | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category           | Task                                   | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Visual gen         | Put Eggplant into Pot (Easy Version)   | 10                | 1                  | 5                | 7                  | 10                         |
| Visual gen         | Put Eggplant into Pot (Easy Version)   | 10                | 1                  | 5                | 7                  | 10                         |
| Visual gen         | Put Eggplant into Pot (Easy Version)   | 10                | 1                  | 5                | 7                  | 10                         |
| Visual gen         | Put Eggplant into Pot (Easy Version)   | 10                | 1                  | 5                | 7                  | 10                         |
| Visual gen         | Put Eggplant into Pot (Easy Version)   | 10                | 1                  | 5                | 7                  | 10                         |
| Visual gen         | Put Eggplant into Pot (Easy Version)   | 10                | 1                  | 5                | 7                  | 10                         |
| Visual gen         | Put Eggplant into Pot (Easy Version)   | 10                | 1                  | 5                | 7                  | 10                         |
| Visual gen         | Put Eggplant into Pot                  | 10                | 0                  | 1                | 5                  | 10                         |
| Visual gen         | Put Eggplant into Pot                  | 10                | 0                  | 1                | 5                  | 10                         |
| Visual gen         | Put Eggplant into Pot                  | 10                | 0                  | 1                | 5                  | 10                         |
| Visual gen         | Put Eggplant into Pot                  | 10                | 0                  | 1                | 5                  | 10                         |
| Visual gen         | Put Eggplant into Pot                  | 10                | 0                  | 1                | 5                  | 10                         |
| Visual gen         | Put Eggplant into Pot                  | 10                | 0                  | 1                | 5                  | 10                         |
| Visual gen         | Put Eggplant into Pot                  | 10                | 0                  | 1                | 5                  | 10                         |
| Visual gen         | Put Cup from Counter into Sink         | 10                | 1                  | 1                | 0                  | 7                          |
| Visual gen         | Put Cup from Counter into Sink         | 10                | 1                  | 1                | 0                  | 7                          |
| Visual gen         | Put Cup from Counter into Sink         | 10                | 1                  | 1                | 0                  | 7                          |
| Visual gen         | Put Cup from Counter into Sink         | 10                | 1                  | 1                | 0                  | 7                          |
| Visual gen         | Put Cup from Counter into Sink         | 10                | 1                  | 1                | 0                  | 7                          |
| Visual gen         | Put Cup from Counter into Sink         | 10                | 1                  | 1                | 0                  | 7                          |
| Visual gen         | Put Cup from Counter into Sink         | 10                | 1                  | 1                | 0                  | 7                          |
| Visual gen         | Put Eggplant into Pot (w/ Clutter)     | 10                | 1                  | 3.5              | 6                  | 7.5                        |
| Visual gen         | Put Eggplant into Pot (w/ Clutter)     | 10                | 1                  | 3.5              | 6                  | 7.5                        |
| Visual gen         | Put Eggplant into Pot (w/ Clutter)     | 10                | 1                  | 3.5              | 6                  | 7.5                        |
| Visual gen         | Put Eggplant into Pot (w/ Clutter)     | 10                | 1                  | 3.5              | 6                  | 7.5                        |
| Visual gen         | Put Eggplant into Pot (w/ Clutter)     | 10                | 1                  | 3.5              | 6                  | 7.5                        |
| Visual gen         | Put Eggplant into Pot (w/ Clutter)     | 10                | 1                  | 3.5              | 6                  | 7.5                        |
| Visual gen         | Put Eggplant into Pot (w/ Clutter)     | 10                | 1                  | 3.5              | 6                  | 7.5                        |
| Visual gen         | Put Yellow Corn on Pink Plate          | 10                | 1                  | 4                | 8                  | 9                          |
| Visual gen         | Put Yellow Corn on Pink Plate          | 10                | 1                  | 4                | 8                  | 9                          |
| Visual gen         | Put Yellow Corn on Pink Plate          | 10                | 1                  | 4                | 8                  | 9                          |
| Visual gen         | Put Yellow Corn on Pink Plate          | 10                | 1                  | 4                | 8                  | 9                          |
| Visual gen         | Put Yellow Corn on Pink Plate          | 10                | 1                  | 4                | 8                  | 9                          |
| Visual gen         | Put Yellow Corn on Pink Plate          | 10                | 1                  | 4                | 8                  | 9                          |
| Visual gen         | Put Yellow Corn on Pink Plate          | 10                | 1                  | 4                | 8                  | 9                          |
| Motion gen         | Lift Eggplant                          | 10                | 3                  | 0.5              | 6.5                | 7.5                        |
| Motion gen         | Lift Eggplant                          | 10                | 3                  | 0.5              | 6.5                | 7.5                        |
| Motion gen         | Lift Eggplant                          | 10                | 3                  | 0.5              | 6.5                | 7.5                        |
| Motion gen         | Lift Eggplant                          | 10                | 3                  | 0.5              | 6.5                | 7.5                        |
| Motion gen         | Lift Eggplant                          | 10                | 3                  | 0.5              | 6.5                | 7.5                        |
| Motion gen         | Lift Eggplant                          | 10                | 3                  | 0.5              | 6.5                | 7.5                        |
| Motion gen         | Lift Eggplant                          | 10                | 3                  | 0.5              | 6.5                | 7.5                        |
| Motion gen         | Put Carrot on Plate (w/ Height Change) | 10                | 2                  | 1                | 4.5                | 4.5                        |
| Motion gen         | Put Carrot on Plate (w/ Height Change) | 10                | 2                  | 1                | 4.5                | 4.5                        |
| Motion gen         | Put Carrot on Plate (w/ Height Change) | 10                | 2                  | 1                | 4.5                | 4.5                        |
| Motion gen         | Put Carrot on Plate (w/ Height Change) | 10                | 2                  | 1                | 4.5                | 4.5                        |
| Motion gen         | Put Carrot on Plate (w/ Height Change) | 10                | 2                  | 1                | 4.5                | 4.5                        |
| Motion gen         | Put Carrot on Plate (w/ Height Change) | 10                | 2                  | 1                | 4.5                | 4.5                        |
| Motion gen         | Put Carrot on Plate (w/ Height Change) | 10                | 2                  | 1                | 4.5                | 4.5                        |
| Physical gen       | Put Carrot on Plate                    | 10                | 1                  | 0                | 1                  | 8                          |
| Physical gen       | Put Carrot on Plate                    | 10                | 1                  | 0                | 1                  | 8                          |
| Physical gen       | Put Carrot on Plate                    | 10                | 1                  | 0                | 1                  | 8                          |
| Physical gen       | Put Carrot on Plate                    | 10                | 1                  | 0                | 1                  | 8                          |
| Physical gen       | Put Carrot on Plate                    | 10                | 1                  | 0                | 1                  | 8                          |
| Physical gen       | Put Carrot on Plate                    | 10                | 1                  | 0                | 1                  | 8                          |
| Physical gen       | Put Carrot on Plate                    | 10                | 1                  | 0                | 1                  | 8                          |
| Physical gen       | Flip Pot Upright                       | 10                | 2                  | 6                | 5                  | 8                          |
| Physical gen       | Flip Pot Upright                       | 10                | 2                  | 6                | 5                  | 8                          |
| Physical gen       | Flip Pot Upright                       | 10                | 2                  | 6                | 5                  | 8                          |
| Physical gen       | Flip Pot Upright                       | 10                | 2                  | 6                | 5                  | 8                          |
| Physical gen       | Flip Pot Upright                       | 10                | 2                  | 6                | 5                  | 8                          |
| Physical gen       | Flip Pot Upright                       | 10                | 2                  | 6                | 5                  | 8                          |
| Physical gen       | Flip Pot Upright                       | 10                | 2                  | 6                | 5                  | 8                          |
| Physical gen       | Lift AAA Battery                       | 10                | 0                  | 0                | 2                  | 7                          |
| Physical gen       | Lift AAA Battery                       | 10                | 0                  | 0                | 2                  | 7                          |
| Physical gen       | Lift AAA Battery                       | 10                | 0                  | 0                | 2                  | 7                          |
| Physical gen       | Lift AAA Battery                       | 10                | 0                  | 0                | 2                  | 7                          |
| Physical gen       | Lift AAA Battery                       | 10                | 0                  | 0                | 2                  | 7                          |
| Physical gen       | Lift AAA Battery                       | 10                | 0                  | 0                | 2                  | 7                          |
| Physical gen       | Lift AAA Battery                       | 10                | 0                  | 0                | 2                  | 7                          |
| Semantic gen       | Move Skull into Drying Rack            | 10                | 1                  | 0                | 5                  | 5                          |
| Semantic gen       | Move Skull into Drying Rack            | 10                | 1                  | 0                | 5                  | 5                          |
| Semantic gen       | Move Skull into Drying Rack            | 10                | 1                  | 0                | 5                  | 5                          |
| Semantic gen       | Move Skull into Drying Rack            | 10                | 1                  | 0                | 5                  | 5                          |
| Semantic gen       | Move Skull into Drying Rack            | 10                | 1                  | 0                | 5                  | 5                          |
| Semantic gen       | Move Skull into Drying Rack            | 10                | 1                  | 0                | 5                  | 5                          |
| Semantic gen       | Move Skull into Drying Rack            | 10                | 1                  | 0                | 5                  | 5                          |
| Semantic gen       | Lift White Tape                        | 10                | 3                  | 0                | 0                  | 1                          |
| Semantic gen       | Lift White Tape                        | 10                | 3                  | 0                | 0                  | 1                          |
| Semantic gen       | Lift White Tape                        | 10                | 3                  | 0                | 0                  | 1                          |
| Semantic gen       | Lift White Tape                        | 10                | 3                  | 0                | 0                  | 1                          |
| Semantic gen       | Lift White Tape                        | 10                | 3                  | 0                | 0                  | 1                          |
| Semantic gen       | Lift White Tape                        | 10                | 3                  | 0                | 0                  | 1                          |
| Semantic gen       | Lift White Tape                        | 10                | 3                  | 0                | 0                  | 1                          |
| Semantic gen       | Take Purple Grapes out of Pot          | 10                | 6                  | 0                | 5                  | 4                          |
| Semantic gen       | Take Purple Grapes out of Pot          | 10                | 6                  | 0                | 5                  | 4                          |
| Semantic gen       | Take Purple Grapes out of Pot          | 10                | 6                  | 0                | 5                  | 4                          |
| Semantic gen       | Take Purple Grapes out of Pot          | 10                | 6                  | 0                | 5                  | 4                          |
| Semantic gen       | Take Purple Grapes out of Pot          | 10                | 6                  | 0                | 5                  | 4                          |
| Semantic gen       | Take Purple Grapes out of Pot          | 10                | 6                  | 0                | 5                  | 4                          |
| Semantic gen       | Take Purple Grapes out of Pot          | 10                | 6                  | 0                | 5                  | 4                          |
| Semantic gen       | Stack Blue Cup on Pink Cup             | 10                | 0.5                | 0                | 5.5                | 4.5                        |
| Semantic gen       | Stack Blue Cup on Pink Cup             | 10                | 0.5                | 0                | 5.5                | 4.5                        |
| Semantic gen       | Stack Blue Cup on Pink Cup             | 10                | 0.5                | 0                | 5.5                | 4.5                        |
| Semantic gen       | Stack Blue Cup on Pink Cup             | 10                | 0.5                | 0                | 5.5                | 4.5                        |
| Semantic gen       | Stack Blue Cup on Pink Cup             | 10                | 0.5                | 0                | 5.5                | 4.5                        |
| Semantic gen       | Stack Blue Cup on Pink Cup             | 10                | 0.5                | 0                | 5.5                | 4.5                        |
| Semantic gen       | Stack Blue Cup on Pink Cup             | 10                | 0.5                | 0                | 5.5                | 4.5                        |
| Language grounding | Put {Eggplant, Red Bottle} into Pot    | 10                | 2.5                | 4                | 8.5                | 7.5                        |
| Language grounding | Put {Eggplant, Red Bottle} into Pot    | 10                | 2.5                | 4                | 8.5                | 7.5                        |
| Language grounding | Put {Eggplant, Red Bottle} into Pot    | 10                | 2.5                | 4                | 8.5                | 7.5                        |
| Language grounding | Put {Eggplant, Red Bottle} into Pot    | 10                | 2.5                | 4                | 8.5                | 7.5                        |
| Language grounding | Put {Eggplant, Red Bottle} into Pot    | 10                | 2.5                | 4                | 8.5                | 7.5                        |
| Language grounding | Put {Eggplant, Red Bottle} into Pot    | 10                | 2.5                | 4                | 8.5                | 7.5                        |
| Language grounding | Put {Eggplant, Red Bottle} into Pot    | 10                | 2.5                | 4                | 8.5                | 7.5                        |
| Language grounding | Lift {Cheese, Red Chili Pepper}        | 10                | 1.5                | 2.5              | 8.5                | 10                         |
| Language grounding | Lift {Cheese, Red Chili Pepper}        | 10                | 1.5                | 2.5              | 8.5                | 10                         |
| Language grounding | Lift {Cheese, Red Chili Pepper}        | 10                | 1.5                | 2.5              | 8.5                | 10                         |
| Language grounding | Lift {Cheese, Red Chili Pepper}        | 10                | 1.5                | 2.5              | 8.5                | 10                         |
| Language grounding | Lift {Cheese, Red Chili Pepper}        | 10                | 1.5                | 2.5              | 8.5                | 10                         |
| Language grounding | Lift {Cheese, Red Chili Pepper}        | 10                | 1.5                | 2.5              | 8.5                | 10                         |
| Language grounding | Lift {Cheese, Red Chili Pepper}        | 10                | 1.5                | 2.5              | 8.5                | 10                         |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate      | 10                | 5                  | 5.5              | 8.5                | 9.5                        |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate      | 10                | 5                  | 5.5              | 8.5                | 9.5                        |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate      | 10                | 5                  | 5.5              | 8.5                | 9.5                        |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate      | 10                | 5                  | 5.5              | 8.5                | 9.5                        |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate      | 10                | 5                  | 5.5              | 8.5                | 9.5                        |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate      | 10                | 5                  | 5.5              | 8.5                | 9.5                        |
| Language grounding | Put {Blue Cup, Pink Cup} on Plate      | 10                | 5                  | 5.5              | 8.5                | 9.5                        |
|                    |                                        | Mean Success Rate | 18.5$\pm$2.7%      | 20.0$\pm$2.6%    | 50.6$\pm$3.5%      | 70.6$\pm$3.2%              |
|                    |                                        | Mean Success Rate | 18.5$\pm$2.7%      | 20.0$\pm$2.6%    | 50.6$\pm$3.5%      | 70.6$\pm$3.2%              |
|                    |                                        | Mean Success Rate | 18.5$\pm$2.7%      | 20.0$\pm$2.6%    | 50.6$\pm$3.5%      | 70.6$\pm$3.2%              |
|                    |                                        | Mean Success Rate | 18.5$\pm$2.7%      | 20.0$\pm$2.6%    | 50.6$\pm$3.5%      | 70.6$\pm$3.2%              |
|                    |                                        | Mean Success Rate | 18.5$\pm$2.7%      | 20.0$\pm$2.6%    | 50.6$\pm$3.5%      | 70.6$\pm$3.2%              |
|                    |                                        | Mean Success Rate | 18.5$\pm$2.7%      | 20.0$\pm$2.6%    | 50.6$\pm$3.5%      | 70.6$\pm$3.2%              |
|                    |                                        | Mean Success Rate | 18.5$\pm$2.7%      | 20.0$\pm$2.6%    | 50.6$\pm$3.5%      | 70.6$\pm$3.2%              |

Additionally, in [Table 5](#table-5), we provide the full evaluation results for the quantized inference experiments that were summarized in [Table 2](#table-2). For these evaluations, we test policies on 8 representative BridgeData V2 tasks spanning all task categories in the full evaluation suite.

<a id="table-5"></a>

> Table 5: Full quantized inference results. Here we present the detailed version of the results shown in [Table 2](#table-2).

| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| ------------------ | ------------------------------------ | ----------------- | -------------------- | ---------------- | ---------------- |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 9                    | 7                | 9                |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 9                    | 7                | 9                |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 9                    | 7                | 9                |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 9                    | 7                | 9                |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 9                    | 7                | 9                |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 9                    | 7                | 9                |
| Visual gen         | Put Eggplant into Pot                | 10                | 7                    | 7                | 7                |
| Visual gen         | Put Eggplant into Pot                | 10                | 7                    | 7                | 7                |
| Visual gen         | Put Eggplant into Pot                | 10                | 7                    | 7                | 7                |
| Visual gen         | Put Eggplant into Pot                | 10                | 7                    | 7                | 7                |
| Visual gen         | Put Eggplant into Pot                | 10                | 7                    | 7                | 7                |
| Visual gen         | Put Eggplant into Pot                | 10                | 7                    | 7                | 7                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 3                | 7                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 3                | 7                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 3                | 7                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 3                | 7                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 3                | 7                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 3                | 7                |
| Motion gen         | Lift Eggplant                        | 10                | 6                    | 4                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 6                    | 4                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 6                    | 4                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 6                    | 4                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 6                    | 4                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 6                    | 4                | 7.5              |
| Physical gen       | Put Carrot on Plate                  | 10                | 6                    | 5                | 7                |
| Physical gen       | Put Carrot on Plate                  | 10                | 6                    | 5                | 7                |
| Physical gen       | Put Carrot on Plate                  | 10                | 6                    | 5                | 7                |
| Physical gen       | Put Carrot on Plate                  | 10                | 6                    | 5                | 7                |
| Physical gen       | Put Carrot on Plate                  | 10                | 6                    | 5                | 7                |
| Physical gen       | Put Carrot on Plate                  | 10                | 6                    | 5                | 7                |
| Physical gen       | Lift AAA Battery                     | 10                | 7                    | 5                | 3                |
| Physical gen       | Lift AAA Battery                     | 10                | 7                    | 5                | 3                |
| Physical gen       | Lift AAA Battery                     | 10                | 7                    | 5                | 3                |
| Physical gen       | Lift AAA Battery                     | 10                | 7                    | 5                | 3                |
| Physical gen       | Lift AAA Battery                     | 10                | 7                    | 5                | 3                |
| Physical gen       | Lift AAA Battery                     | 10                | 7                    | 5                | 3                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 8                    | 8                | 9                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 8                    | 8                | 9                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 8                    | 8                | 9                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 8                    | 8                | 9                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 8                    | 8                | 9                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 8                    | 8                | 9                |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 7.5              | 8                |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 7.5              | 8                |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 7.5              | 8                |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 7.5              | 8                |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 7.5              | 8                |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 7.5              | 8                |
|                    |                                      | Mean Success Rate | 71.3 $\pm$ 4.8%      | 58.1 $\pm$ 5.1%  | 71.9 $\pm$ 4.7%  |
|                    |                                      | Mean Success Rate | 71.3 $\pm$ 4.8%      | 58.1 $\pm$ 5.1%  | 71.9 $\pm$ 4.7%  |
|                    |                                      | Mean Success Rate | 71.3 $\pm$ 4.8%      | 58.1 $\pm$ 5.1%  | 71.9 $\pm$ 4.7%  |
|                    |                                      | Mean Success Rate | 71.3 $\pm$ 4.8%      | 58.1 $\pm$ 5.1%  | 71.9 $\pm$ 4.7%  |
|                    |                                      | Mean Success Rate | 71.3 $\pm$ 4.8%      | 58.1 $\pm$ 5.1%  | 71.9 $\pm$ 4.7%  |
|                    |                                      | Mean Success Rate | 71.3 $\pm$ 4.8%      | 58.1 $\pm$ 5.1%  | 71.9 $\pm$ 4.7%  |

### B.2 Google Robot Evaluation Details

In this section, we provide more details on the Google robot evaluations introduced in [Section 5.1](#section-5-1).

- [B.2.1 Google Robot Evaluation Tasks](#b21-google-robot-evaluation-tasks)
- [B.2.2 Detailed Google Robot Evaluation Results](#b22-detailed-google-robot-evaluation-results)

#### B.2.1 Google Robot Evaluation Tasks

<a id="figure-8"></a>

![rt1_robot_tasks](images/rt1_robot_tasks.jpeg)

> Figure 8: Google robot evaluation tasks. We evaluate every generalist robot policy on in-distribution tasks and out-of-distribution (OOD) generalization tasks. OOD tasks involve unseen backgrounds, target objects, instructions/object relations, and semantic concepts (e.g., photos from the Internet that do not appear in robot action data).

On the Google robot, we evaluate each generalist robot policy on 12 tasks with 5 rollouts each, for a total of 60 rollouts. The first five tasks test on in-distribution conditions, and the last seven tasks test on more difficult out-of-distribution (OOD) conditions. All tasks are depicted in [Fig. 8](https://arxiv.org/html/2406.09246v3#A2.F8). Each rollout is marked as a failure (0) or success (1).

We describe the 12 tasks below:

- 1. Pick Coke Can (in-distribution): The robot is positioned in front of a platform with a can of Coke on top of it. The robot’s goal is to grasp and lift the Coke can.
- 2. Move Apple near Green Can (in-distribution): The robot is positioned in front of a platform with an apple and a green soda can on top of it. The robot’s goal is to grasp the apple and move it next to the green can.
- 3. Move Blue Chip Bag near Apple (in-distribution): The robot is positioned in front of a platform with a blue bag of chips and an apple on top of it. The robot’s goal is to grasp the blue bag of chips and move it close to the apple.
- 4. Place Coke Can Upright (in-distribution): The robot is positioned in front of a platform with a can of Coke on top of it, and the can is oriented horizontally on its side. The robot’s goal is to grasp the Coke can and orient it to be in a vertical position.
- 5. Open Middle Drawer (in-distribution): The robot is positioned in front of a set of three drawers. The robot’s goal is to grasp the middle drawer handle and pull the drawer open.
- 6. Move Orange near Brown Chip Bag (OOD): The robot is positioned in front of a platform with a brown bag of chips and an orange on top of it. A tablecloth with blue sky and white cloud patterns covers the platform underneath the objects. The robot’s goal is to grasp the orange and bring it next to the bag of chips. This task is OOD because the orange is an unseen object relative to the training dataset, and the tablecloth is an unseen background.(^7^77See Appendix of Brohan et al. [[7]] for a detailed list of OOD conditions in Google robot evaluations.)
- 7. Pick Pepsi Can (OOD): The robot is positioned in front of a platform with a can of Pepsi on top of it. A tablecloth with bright yellow/brown patterns covers the platform underneath the can. The robot’s goal is to grasp and lift the can. This task is OOD because the Pepsi can is an unseen object, and the tablecloth is an unseen background.
- 8. Pick Banana (OOD): The robot is positioned in front of a platform with an apple, a can of Coke, and a banana. The robot’s goal is to grasp and lift the banana. This task is OOD because the banana is an unseen target object.
- 9. Pick Green Cup (OOD): The robot is positioned in front of a platform with a banana, a can of Pepsi, and a green cup. The robot’s goal is to grasp and lift the green cup. This task is OOD because all objects in the scene are unseen in the training data.
- 10. Place Apple on Plate (OOD): The robot is positioned in front of a platform with a plate and an apple. The robot’s goal is to grasp the apple and move it onto the plate. This task is OOD because it is a novel instruction describing an unseen object relation: training demonstrations only cover moving the apple _near_ the plate, rather than placing it _on top of_ the plate.
- 11. Place Banana in Pan (OOD): The robot is positioned in front of a platform with a pan and a banana. The robot’s goal is to grasp the banana and move it into the pan. This task is OOD because the banana is an unseen target object, and it is a novel instruction describing an unseen object relation, as explained in the previous task.
- 12. Move Coke Can to Taylor Swift (OOD): The robot is positioned in front of a platform with a can of Coke and photos of three different celebrities, including Taylor Swift. The robot’s goal is to grasp the can and move it to the photo of Taylor Swift. This task is OOD because the photos of the celebrities are unseen in the robot interaction data.

#### B.2.2 Detailed Google Robot Evaluation Results

<a id="table-6"></a>

> Table 6: Detailed Google robot evaluation results. We report full evaluation results for Google robot evaluations discussed in [Section 5.1](#section-5-1). Each generalist policy is evaluated with 60 rollouts across 12 tasks, covering both in-distribution and out-of-distribution (OOD) testing conditions. In the bottom row, we report mean success rate $\pm$ StdErr for each policy. OpenVLA and RT-2-X both significantly outperform RT-1-X and Octo overall (we bold the mean success rate for both due to overlapping error bars). See [Fig. 8](https://arxiv.org/html/2406.09246v3#A2.F8) for illustrations of all tasks.

| Category        | Task                            | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| --------------- | ------------------------------- | ----------------- | ------------------ | ---------------- | ------------------ | -------------------------- |
| Category        | Task                            | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category        | Task                            | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category        | Task                            | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category        | Task                            | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category        | Task                            | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| Category        | Task                            | # Trials          | RT-1-X # Successes | Octo # Successes | RT-2-X # Successes | OpenVLA (ours) # Successes |
| In-distribution | Pick Coke Can                   | 5                 | 5                  | 1                | 5                  | 5                          |
| In-distribution | Pick Coke Can                   | 5                 | 5                  | 1                | 5                  | 5                          |
| In-distribution | Pick Coke Can                   | 5                 | 5                  | 1                | 5                  | 5                          |
| In-distribution | Pick Coke Can                   | 5                 | 5                  | 1                | 5                  | 5                          |
| In-distribution | Pick Coke Can                   | 5                 | 5                  | 1                | 5                  | 5                          |
| In-distribution | Pick Coke Can                   | 5                 | 5                  | 1                | 5                  | 5                          |
| In-distribution | Pick Coke Can                   | 5                 | 5                  | 1                | 5                  | 5                          |
| In-distribution | Move Apple near Green Can       | 5                 | 3                  | 3                | 3                  | 5                          |
| In-distribution | Move Apple near Green Can       | 5                 | 3                  | 3                | 3                  | 5                          |
| In-distribution | Move Apple near Green Can       | 5                 | 3                  | 3                | 3                  | 5                          |
| In-distribution | Move Apple near Green Can       | 5                 | 3                  | 3                | 3                  | 5                          |
| In-distribution | Move Apple near Green Can       | 5                 | 3                  | 3                | 3                  | 5                          |
| In-distribution | Move Apple near Green Can       | 5                 | 3                  | 3                | 3                  | 5                          |
| In-distribution | Move Apple near Green Can       | 5                 | 3                  | 3                | 3                  | 5                          |
| In-distribution | Move Blue Chip Bag near Apple   | 5                 | 0                  | 3                | 4                  | 5                          |
| In-distribution | Move Blue Chip Bag near Apple   | 5                 | 0                  | 3                | 4                  | 5                          |
| In-distribution | Move Blue Chip Bag near Apple   | 5                 | 0                  | 3                | 4                  | 5                          |
| In-distribution | Move Blue Chip Bag near Apple   | 5                 | 0                  | 3                | 4                  | 5                          |
| In-distribution | Move Blue Chip Bag near Apple   | 5                 | 0                  | 3                | 4                  | 5                          |
| In-distribution | Move Blue Chip Bag near Apple   | 5                 | 0                  | 3                | 4                  | 5                          |
| In-distribution | Move Blue Chip Bag near Apple   | 5                 | 0                  | 3                | 4                  | 5                          |
| In-distribution | Place Coke Can Upright          | 5                 | 0                  | 0                | 4                  | 4                          |
| In-distribution | Place Coke Can Upright          | 5                 | 0                  | 0                | 4                  | 4                          |
| In-distribution | Place Coke Can Upright          | 5                 | 0                  | 0                | 4                  | 4                          |
| In-distribution | Place Coke Can Upright          | 5                 | 0                  | 0                | 4                  | 4                          |
| In-distribution | Place Coke Can Upright          | 5                 | 0                  | 0                | 4                  | 4                          |
| In-distribution | Place Coke Can Upright          | 5                 | 0                  | 0                | 4                  | 4                          |
| In-distribution | Place Coke Can Upright          | 5                 | 0                  | 0                | 4                  | 4                          |
| In-distribution | Open Middle Drawer              | 5                 | 0                  | 4                | 2                  | 3                          |
| In-distribution | Open Middle Drawer              | 5                 | 0                  | 4                | 2                  | 3                          |
| In-distribution | Open Middle Drawer              | 5                 | 0                  | 4                | 2                  | 3                          |
| In-distribution | Open Middle Drawer              | 5                 | 0                  | 4                | 2                  | 3                          |
| In-distribution | Open Middle Drawer              | 5                 | 0                  | 4                | 2                  | 3                          |
| In-distribution | Open Middle Drawer              | 5                 | 0                  | 4                | 2                  | 3                          |
| In-distribution | Open Middle Drawer              | 5                 | 0                  | 4                | 2                  | 3                          |
| OOD             | Move Orange near Brown Chip Bag | 5                 | 1                  | 2                | 5                  | 5                          |
| OOD             | Move Orange near Brown Chip Bag | 5                 | 1                  | 2                | 5                  | 5                          |
| OOD             | Move Orange near Brown Chip Bag | 5                 | 1                  | 2                | 5                  | 5                          |
| OOD             | Move Orange near Brown Chip Bag | 5                 | 1                  | 2                | 5                  | 5                          |
| OOD             | Move Orange near Brown Chip Bag | 5                 | 1                  | 2                | 5                  | 5                          |
| OOD             | Move Orange near Brown Chip Bag | 5                 | 1                  | 2                | 5                  | 5                          |
| OOD             | Move Orange near Brown Chip Bag | 5                 | 1                  | 2                | 5                  | 5                          |
| OOD             | Pick Pepsi Can                  | 5                 | 3                  | 0                | 5                  | 4                          |
| OOD             | Pick Pepsi Can                  | 5                 | 3                  | 0                | 5                  | 4                          |
| OOD             | Pick Pepsi Can                  | 5                 | 3                  | 0                | 5                  | 4                          |
| OOD             | Pick Pepsi Can                  | 5                 | 3                  | 0                | 5                  | 4                          |
| OOD             | Pick Pepsi Can                  | 5                 | 3                  | 0                | 5                  | 4                          |
| OOD             | Pick Pepsi Can                  | 5                 | 3                  | 0                | 5                  | 4                          |
| OOD             | Pick Pepsi Can                  | 5                 | 3                  | 0                | 5                  | 4                          |
| OOD             | Pick Banana                     | 5                 | 5                  | 3                | 5                  | 5                          |
| OOD             | Pick Banana                     | 5                 | 5                  | 3                | 5                  | 5                          |
| OOD             | Pick Banana                     | 5                 | 5                  | 3                | 5                  | 5                          |
| OOD             | Pick Banana                     | 5                 | 5                  | 3                | 5                  | 5                          |
| OOD             | Pick Banana                     | 5                 | 5                  | 3                | 5                  | 5                          |
| OOD             | Pick Banana                     | 5                 | 5                  | 3                | 5                  | 5                          |
| OOD             | Pick Banana                     | 5                 | 5                  | 3                | 5                  | 5                          |
| OOD             | Pick Green Cup                  | 5                 | 1                  | 0                | 5                  | 5                          |
| OOD             | Pick Green Cup                  | 5                 | 1                  | 0                | 5                  | 5                          |
| OOD             | Pick Green Cup                  | 5                 | 1                  | 0                | 5                  | 5                          |
| OOD             | Pick Green Cup                  | 5                 | 1                  | 0                | 5                  | 5                          |
| OOD             | Pick Green Cup                  | 5                 | 1                  | 0                | 5                  | 5                          |
| OOD             | Pick Green Cup                  | 5                 | 1                  | 0                | 5                  | 5                          |
| OOD             | Pick Green Cup                  | 5                 | 1                  | 0                | 5                  | 5                          |
| OOD             | Place Apple on Plate            | 5                 | 0                  | 0                | 4                  | 4                          |
| OOD             | Place Apple on Plate            | 5                 | 0                  | 0                | 4                  | 4                          |
| OOD             | Place Apple on Plate            | 5                 | 0                  | 0                | 4                  | 4                          |
| OOD             | Place Apple on Plate            | 5                 | 0                  | 0                | 4                  | 4                          |
| OOD             | Place Apple on Plate            | 5                 | 0                  | 0                | 4                  | 4                          |
| OOD             | Place Apple on Plate            | 5                 | 0                  | 0                | 4                  | 4                          |
| OOD             | Place Apple on Plate            | 5                 | 0                  | 0                | 4                  | 4                          |
| OOD             | Place Banana in Pan             | 5                 | 0                  | 0                | 2                  | 4                          |
| OOD             | Place Banana in Pan             | 5                 | 0                  | 0                | 2                  | 4                          |
| OOD             | Place Banana in Pan             | 5                 | 0                  | 0                | 2                  | 4                          |
| OOD             | Place Banana in Pan             | 5                 | 0                  | 0                | 2                  | 4                          |
| OOD             | Place Banana in Pan             | 5                 | 0                  | 0                | 2                  | 4                          |
| OOD             | Place Banana in Pan             | 5                 | 0                  | 0                | 2                  | 4                          |
| OOD             | Place Banana in Pan             | 5                 | 0                  | 0                | 2                  | 4                          |
| OOD             | Move Coke Can near Taylor Swift | 5                 | 2                  | 0                | 3                  | 2                          |
| OOD             | Move Coke Can near Taylor Swift | 5                 | 2                  | 0                | 3                  | 2                          |
| OOD             | Move Coke Can near Taylor Swift | 5                 | 2                  | 0                | 3                  | 2                          |
| OOD             | Move Coke Can near Taylor Swift | 5                 | 2                  | 0                | 3                  | 2                          |
| OOD             | Move Coke Can near Taylor Swift | 5                 | 2                  | 0                | 3                  | 2                          |
| OOD             | Move Coke Can near Taylor Swift | 5                 | 2                  | 0                | 3                  | 2                          |
| OOD             | Move Coke Can near Taylor Swift | 5                 | 2                  | 0                | 3                  | 2                          |
|                 |                                 | Mean Success Rate | 33.3$\pm$6.1%      | 26.7$\pm$5.8%    | 78.3$\pm$5.4%      | 85.0$\pm$4.6%              |
|                 |                                 | Mean Success Rate | 33.3$\pm$6.1%      | 26.7$\pm$5.8%    | 78.3$\pm$5.4%      | 85.0$\pm$4.6%              |
|                 |                                 | Mean Success Rate | 33.3$\pm$6.1%      | 26.7$\pm$5.8%    | 78.3$\pm$5.4%      | 85.0$\pm$4.6%              |
|                 |                                 | Mean Success Rate | 33.3$\pm$6.1%      | 26.7$\pm$5.8%    | 78.3$\pm$5.4%      | 85.0$\pm$4.6%              |
|                 |                                 | Mean Success Rate | 33.3$\pm$6.1%      | 26.7$\pm$5.8%    | 78.3$\pm$5.4%      | 85.0$\pm$4.6%              |
|                 |                                 | Mean Success Rate | 33.3$\pm$6.1%      | 26.7$\pm$5.8%    | 78.3$\pm$5.4%      | 85.0$\pm$4.6%              |
|                 |                                 | Mean Success Rate | 33.3$\pm$6.1%      | 26.7$\pm$5.8%    | 78.3$\pm$5.4%      | 85.0$\pm$4.6%              |

Full results for the Google robot evaluations are shown in [Table 6](#table-6). Overall, we find that RT-1-X and Octo experience difficulty on the evaluation tasks; they are often unable to achieve a single success out of five trials in several tasks. On the other hand, RT-2-X and OpenVLA demonstrate strong performance, completing every task at least two times out of five trials; these two VLA policies perform comparably with each other on this particular evaluation suite.

### B.3 Data-Efficient Adaptation Experiment Details

In this section, we provide more details on the data-efficient adaptation experiments discussed in [Section 5.2](#section-5-2), where we investigate the effectiveness of fine-tuned OpenVLA policies on new robot setups such as Franka-Tabletop and Franka-DROID.

- [B.3.1 Franka-Tabletop and Franka-DROID Tasks](#b31-franka-tabletop-and-franka-droid-tasks)
- [B.3.2 Detailed Franka-Tabletop and Franka-DROID Evaluation Results](#b32-detailed-franka-tabletop-and-franka-droid-evaluation-results)

#### B.3.1 Franka-Tabletop and Franka-DROID Tasks

We collect 10–150 demonstrations of each of seven tasks. The first six tasks correspond to a robot setup which we denote as “Franka-Tabletop” (Franka Emika Panda robot mounted on top of a table), and the final task corresponds to a robot setup which we call “Franka-DROID”.

In the Franka-Tabletop setup, the first three of six tasks correspond to single-instruction tasks and are narrow, while the last three tasks correspond to multi-instruction tasks in which multiple objects are present in the scene and the robot must manipulate the correct one depending on the language instruction.

<a id="figure-9"></a>

![openvla_teaser](images/openvla_teaser.png)

> Figure 9: Franka-Tabletop fine-tuning tasks. Franka-Tabletop tasks used in the data-efficient adaptation experiments in [Section 5.2](#section-5-2) and described in detail in [Fig. 9](https://arxiv.org/html/2406.09246v3#A2.F9) are depicted above. The first three of six tasks, shown in the top three rows, only involve a single instruction, while the last three tasks in the bottom three rows involve multiple objects and instructions (the instructions specify the target object or target location). The first column shows sample initial states matching the training data distribution, while the second column shows out-of-distribution (OOD) initial states (e.g., unseen backgrounds, target objects, distractors, and object positions/orientations). Every policy in [Section 5.2](#section-5-2) is evaluated with 10–12 rollouts on in-distribution tasks and 5–6 rollouts on OOD tasks.

Below we describe each of the six Franka-Tabletop tasks shown in [Fig. 9](https://arxiv.org/html/2406.09246v3#A2.F9):

- 1. Put Carrot in Bowl (single-instruction): The robot’s goal is to grasp the carrot and place it into the bowl. We collect 50 demonstrations of this task for the training dataset, randomly placing the carrot and the bowl at different locations on the table in every episode. The carrot is always initialized on the left side of the bowl. During evaluation, each trial is recorded as a success (1) or failure (0); there is no partial credit.
- 2. Pour Corn into Pot (single-instruction): The robot’s goal is to grasp the red bowl, move towards the steel pot, and pour the contents (a yellow corn) into the pot. We collect 50 demonstrations of this task for the training dataset, randomly placing the bowl and the pot at different locations on the table in every episode. The bowl is always initialized on the right side of the pot. During evaluation, each trial is recorded as a success (1) or failure (0); there is no partial credit.
- 3. Flip Pot Upright (single-instruction): The robot’s goal is to grasp the steel pot (which is initially oriented vertically), rotate it to be in the upright position, and place it back onto the table. We collect only 10 demonstrations of this task for the training dataset, randomly placing the steel pot at various locations within a small section of the table. During evaluation, each trial is recorded as a success (1), failure (0), or partial success (0.5). Partial successes include grasping the pot but not orienting it upright, or knocking it over to the upright position but not carefully guiding it. The robot must release the pot at the end of the episode for full credit.
- 4. Move <object> onto Plate (multi-instruction): The robot’s goal is to grasp one out of three objects (depending on the target specified in the language instruction) and place it on the plate on the right side of the table. We collect 150 demonstrations of this task for the training dataset, randomly placing different combinations of three objects on the table and selecting one as the target. The plate is always initialized on the right side of the table. During evaluation, each trial is recorded as a success (1), failure (0), or partial success (0.5). Partial success is recorded when the first object that the robot makes contact with is the correct target object (i.e., the object specified in the language instruction), but the robot does not complete the task.
- 5. Knock <object> Over (multi-instruction): The robot’s goal is to approach one out of three objects (depending on the target specified in the language instruction) and push it until it falls over. We collect 70 demonstrations of this task for the training dataset, randomly placing different combinations of three objects on the table and selecting one as the target. During evaluation, each trial is recorded as a success (1), failure (0), or partial success (0.5). Partial success is recorded when the first object that the robot makes contact with is the correct target object (i.e., the object specified in the language instruction), but the robot does not complete the task.
- 6. Cover <object> with Towel (multi-instruction): The robot’s goal is to grasp the blue towel and place it on one out of three objects (depending on the target specified in the language instruction). We collect 45 demonstrations of this task for the training dataset, randomly placing different combinations of three objects on the table. During evaluation, each trial is recorded as a success (1), failure (0), or partial success (0.5). Partial success is recorded when the first object that the robot touches with the towel is the correct target object (i.e., the object specified in the language instruction), but the robot does not complete the task (e.g., it drops the towel onto the table instead of on top of the target object). Full credit is given when any part of the towel is resting over the top surface of the target object, i.e., the object does not need to be fully covered.

For every Franka-Tabletop task, we evaluate each method with 10–12 in-distribution trials and 5–6 OOD generalization trials. The in-distribution and OOD test conditions are depicted in [Fig. 9](https://arxiv.org/html/2406.09246v3#A2.F9) (second column).

We describe the OOD test conditions for each of the six tasks below:

- 1. Put Carrot in Bowl (OOD): An eggplant (unseen object) replaces the carrot.
- 2. Pour Corn into Pot (OOD): An unseen brown tablecloth covers the tabletop.
- 3. Flip Pot Upright (OOD): An unseen white tablecloth covers the tabletop
- 4. Move <object> onto Plate (OOD): A set of three unseen objects are placed on the table.
- 5. Knock <object> Over (OOD): Two unseen distractor objects (red plastic cup and brown box) are positioned behind the set of three seen objects.
- 6. Cover <object> with Towel (OOD): The three objects on the table are placed upside-down and at unseen positions.

Finally, in the Franka-DROID environment, we experiment with one task and variants of it: Wipe Table (see [Fig. 10](https://arxiv.org/html/2406.09246v3#A2.F10)). In this task, the robot’s goal is to grab the brush and sweep all three small brown objects into the dustpan. We collect 70 demonstrations for this task for the training dataset, varying the positions of all the objects.

<a id="figure-10"></a>

![droid_wipe_task](images/droid_wipe_task.jpeg)

> Figure 10: Franka-DROID fine-tuning task. The “Wipe Table” task shown here is the final task used in the data-efficient adaptation experiments in [Section 5.2](#section-5-2). The left image shows the initial conditions for an in-distribution trial. The right image shows an out-of-distribution trial in which unseen distractor objects are present on the table. To fully complete the task, the robot must grab the brush and sweep all three objects into the dustpan.

At test time, we evaluate on in-distribution conditions matching the training data ([Fig. 10](https://arxiv.org/html/2406.09246v3#A2.F10), left), as well as out-of-distribution (OOD) conditions in which distractor objects are also present in the scene on the table ([Fig. 10](https://arxiv.org/html/2406.09246v3#A2.F10), right). Since there are various possible outcomes for each trial, we define a scoring rubric as follows: The maximum score for each trial is 2 points. The policy receives the full 2 points if the robot sweeps all three objects into the dustpan. It receives 1 point for successfully sweeping one or two objects into the dustpan. Otherwise, it receives 0 points. We evaluate each policy with 18 in-distribution trials and 12 OOD trials, so each policy receives an aggregate score out of 60 points.

#### B.3.2 Detailed Franka-Tabletop and Franka-DROID Evaluation Results

Full evaluation results for both Franka-Tabletop and Franka-DROID evaluations are shown in [Table 7](#table-7). We evaluate the methods discussed in [Section 5.2](#section-5-2). We find that Diffusion Policy demonstrates strong performance on the single-instruction Franka-Tabletop tasks (e.g., “Put Carrot in Bowl” and “Pour Corn in Pot”), outperforming other methods. However, OpenVLA and Octo achieve higher performance in the more diverse multi-instruction tasks (“Move <object> onto Plate”, “Knock <object> Over”, and “Cover <object> with Towel”). In the Franka-DROID environment, OpenVLA obtains best results. Overall, we find that OpenVLA achieves the highest average performance across both tasks.

<a id="table-7"></a>

> Table 7: Detailed data-efficient adaptation experiment results. Here we present the full breakdown of results summarized in [Fig. 4](#figure-4). We report the performance of Diffusion Policy trained from scratch on new robot tasks, as well as generalist policies fine-tuned on the same data. Each policy is tested against both in-distribution and out-of-distribution (OOD) generalization conditions (see [Fig. 9](https://arxiv.org/html/2406.09246v3#A2.F9) for Franka-Tabletop tasks and [Fig. 10](https://arxiv.org/html/2406.09246v3#A2.F10) for Franka-DROID tasks). We find that no single policy performs best on all tasks: Diffusion Policy achieves high success rates on single-instruction tasks, while OpenVLA and Octo performs well on diverse multi-instruction tasks. In terms of aggregate performance, however, OpenVLA obtains the highest average success rate across both environments.

|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
| --------------------- | --------------------------------------------- | -------- | ---------------- | -------------------------- | ------------- | ----------------- | -------------- |
|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
|                       |                                               | # trials | Diffusion Policy | Diffusion Policy (matched) | Octo          | OpenVLA (scratch) | OpenVLA (ours) |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)        | 10       | 90.0%            | 80.0%                      | 40.0%         | 70.0%             | 70.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Put Carrot in Bowl” (OOD)                    | 5        | 20.0%            | 0.0%                       | 20.0%         | 0.0%              | 40.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (in-distribution)        | 10       | 100.0%           | 90.0%                      | 0.0%          | 10.0%             | 50.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Pour Corn into Pot” (OOD)                    | 5        | 80.0%            | 60.0%                      | 0.0%          | 20.0%             | 60.0%          |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (in-distribution)          | 10       | 100.0%           | 85.0%                      | 40.0%         | 85.0%             | 100.0%         |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Flip Pot Upright” (OOD)                      | 5        | 50.0%            | 20.0%                      | 0.0%          | 40.0%             | 80.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (in-distribution)  | 12       | 25.0%            | 25.0%                      | 41.7%         | 8.3%              | 75.0%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Move <object> onto Plate” (OOD)              | 6        | 8.3%             | 33.3%                      | 8.3%          | 33.3%             | 58.3%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (in-distribution)       | 12       | 33.3%            | 25.0%                      | 83.3%         | 75.0%             | 75.0%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Knock <object> Over” (OOD)                   | 6        | 16.7%            | 16.7%                      | 33.3%         | 58.3%             | 83.3%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (in-distribution) | 12       | 16.7%            | 20.8%                      | 91.7%         | 41.7%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | “Cover <object> with Towel” (OOD)             | 6        | 16.7%            | 33.3%                      | 91.7%         | 50.0%             | 50.0%          |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
|                       | Average                                       |          | 48.5$\pm$4.9%    | 43.4$\pm$4.7%              | 43.4$\pm$4.4% | 43.4$\pm$4.6%     | 67.2$\pm$4.0%  |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
| Franka-DROID (15Hz)   | “Wipe Table” (in-distribution)                | 18       | 50.0%            | 27.8%                      | 52.8%         | 25.0%             | 55.6%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | “Wipe Table” + Distractors (OOD)              | 12       | 12.5%            | 25.0%                      | 16.7%         | 16.7%             | 62.5%          |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |
|                       | Average                                       |          | 35.0$\pm$8.0%    | 26.7$\pm$7.5%              | 38.3$\pm$8.5% | 21.7$\pm$6.6%     | 58.3$\pm$7.2%  |

Additionally, in [Table 8](#table-8), we show the detailed version of the parameter-efficient fine-tuning experiment results summarized in [Table 1](#table-1). In these experiments, we use a representative subset of two Franka-Tabletop tasks, with both in-distribution and OOD variants: one narrow single-instruction task (“Put Carrot in Bowl”) and one diverse multi-instruction task (“Move <object> onto Plate”). We use the same number of training demonstrations used in [Section 5.2](#section-5-2) (50 and 150, respectively), which is delineated in [Section B.3.1](https://arxiv.org/html/2406.09246v3#A2.SS3.SSS1).

<a id="table-8"></a>

> Table 8: Detailed parameter-efficient fine-tuning experiment results. Here we present the detailed task performance results summarized in [Table 1](#table-1).

|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
| --------------------- | -------------------------------------------- | -------- | ------------- | --------------- | ------------- | ------------- | ------------- | ------------- |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
|                       |                                              | # trials | Full FT       | Last layer only | Frozen vision | Sandwich      | LoRA, r=32    | LoRA, r=64    |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
| Franka-Tabletop (5Hz) | “Put Carrot in Bowl” (in-distribution)       | 10       | 90.0          | 40.0            | 40.0          | 90.0          | 60.0          | 90.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Put Carrot in Bowl” (OOD)                   | 5        | 40.0          | 0.0             | 40.0          | 0.0           | 60.0          | 40.0          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (in-distribution) | 12       | 79.2          | 33.3            | 50.0          | 75.0          | 75.0          | 62.5          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | “Move <object> onto Plate” (OOD)             | 6        | 41.7          | 33.3            | 58.3          | 41.7          | 75.0          | 66.7          |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |
|                       | Average                                      |          | 69.7$\pm$7.2% | 30.3$\pm$6.1%   | 47.0$\pm$6.9% | 62.1$\pm$7.9% | 68.2$\pm$7.5% | 68.2$\pm$7.8% |

<a id="appendix-c"></a>

## Appendix C RT-2-X vs. OpenVLA in BridgeData V2 Evaluations

In this section, we provide additional details on RT-2-X vs. OpenVLA comparisons in BridgeData V2 evaluations discussed in [Section 5.1](#section-5-1). As discussed previously, OpenVLA is pretrained on a larger subset of OpenX data than RT-2-X and uses a fused SigLIP-DinoV2 vision backbone rather than a single visual encoder. However, in addition to these factors, we believe that OpenVLA’s significant improvement upon RT-2-X specifically in BridgeData V2 evaluations (as shown in [Fig. 2](#figure-2)) also stems from more careful preprocessing of the Bridge dataset.

During the development of the OpenVLA model, we discovered that the original version of the BridgeData V2 dataset contained many transitions with all-zero (no-op) actions. For instance, in every demonstration, an all-zero action was recorded as the ground-truth action in the first timestep. Consequently, training a highly expressive VLA model on the original dataset without any data preprocessing led to a policy that frequently predicted all-zero actions and froze during evaluations. Therefore, we simply filtered out the first transition in every demonstration when training the OpenVLA model, and this was sufficient for mitigating the freezing behavior in most cases.

However, the RT-2-X model was trained without such data preprocessing, so it often suffers the aforementioned freezing behavior if deployed out of the box without modifying the model querying procedure – which severely deteriorates rollout performance. Since this is a proprietary model that is infeasible for us to re-train (e.g., with our preprocessed version of the BridgeData V2 dataset), we mitigated this issue by simply querying the _second-most-likely_ action from the model, since the first-most-likely action was often all zeros while the second-most-likely action was not. (Note that this is the same workaround that was applied by the developers of the RT-2-X model for BridgeData V2 evaluations reported in the Open X-Embodiment experiments [[1]].) This workaround led to much stronger RT-2-X performance on BridgeData V2 evaluations – though we believe that it is still suboptimal compared to re-training the model on the preprocessed version of the dataset.

We also tried to _dynamically_ query RT-2-X, i.e., by first sampling the first-most-likely action and then sampling the second-most-likely action if the first one was all zeros. However, we empirically found that dynamic querying led to worse performance than simply querying the second-most-likely action at all times. We hypothesize that this is due to a change in the robot’s dynamics that arises from dynamic querying: pausing in the middle of a trajectory to re-query the model leads to slight interruptions in the robot’s movement due to non-neglible latency in the querying pipeline, and this leads to subtle performance degradation. Therefore, we report the performance of RT-2-X when always querying the second-most-likely action, as done in the Open X-Embodiment project [[1]].

<a id="appendix-d"></a>

## Appendix D Additional Experiments and Ablations

In this section, we conduct several additional experiments to analyze the effects of individual components of the OpenVLA model architecture and training scheme, as well as provide quantitative evidence for claims made in earlier sections of this work. We aim to answer the following questions:

- 1. How important is OpenX training and how does it impact OpenVLA’s performance ([Section D.1](https://arxiv.org/html/2406.09246v3#A4.SS1))?
- 2. What effect does using a fused SigLIP-DinoV2 vision encoder have on OpenVLA’s performance, compared to using a SigLIP-only vision encoder ([Section D.2](https://arxiv.org/html/2406.09246v3#A4.SS2))?
- 3. Is it better to fine-tune or freeze the vision encoder in OpenVLA ([Section D.3](https://arxiv.org/html/2406.09246v3#A4.SS3))?
- 4. How do the quantized inference results discussed in [Section 5.3](#section-5-3) change when policy performance is disentangled from model inference speed ([Section D.4](https://arxiv.org/html/2406.09246v3#A4.SS4))?

We discuss the experimental setup and results addressing each of the above questions sequentially in the following sections.

- [D.1 OpenX Training Data Ablation Experiments](#d1-openx-training-data-ablation-experiments)
- [D.2 Dual vs. Single Vision Encoder Experiments](#d2-dual-vs-single-vision-encoder-experiments)
- [D.3 Fine-Tuned vs. Frozen Vision Encoder Experiments](#d3-fine-tuned-vs-frozen-vision-encoder-experiments)
- [D.4 Additional Quantized Inference Experiments: Disentangling Policy Performance and Model Inference Speed](#d4-additional-quantized-inference-experiments-disentangling-policy-performance-and-model-inference-speed)

### D.1 OpenX Training Data Ablation Experiments

As discussed in [Section 3.3](#section-3-3), OpenVLA is trained on a large dataset of robot embodiments, scenes, and tasks from the Open X-Embodiment dataset [[1]] (OpenX). In this section, we ablate the OpenX mixture and train a VLA policy solely on one robot dataset, to assess the impact of OpenX training on policy performance. Note that we have already observed the negative effect of ablating OpenX training in the fine-tuning regime, as discussed in [Section 5.2](#section-5-2) (see OpenVLA (Scratch)), but we discuss additional experiments on another robot embodiment in this section to provide more supporting evidence.

Experimental setup and tasks. We compare the original OpenVLA model with OpenVLA-Bridge, which is produced by taking the same pretrained VLM as OpenVLA (Prismatic VLM [[44]]) and fine-tuning it solely on BridgeData V2 [[6]] rather than the entire OpenX training mixture discussed in [Appendix A](#appendix-a). We evaluate OpenVLA and OpenVLA-Bridge on a subset of 8 representative tasks from the BridgeData V2 WidowX robot evaluation suite discussed in [Section B.1.1](https://arxiv.org/html/2406.09246v3#A2.SS1.SSS1). The tasks are listed in [Table 9](#table-9).

Results. Results for the OpenX training mixture ablation are shown in [Table 9](#table-9). By comparing OpenVLA with OpenVLA-Bridge, we see that performance drops drastically (reduction of 30 percent in absolute success rate), which demonstrates the importance of OpenX pretraining on final policy performance. Although the language grounding performance is not impacted, we observe performance reduction across all generalization categories. This result suggests that the large diversity of scenes, objects, and tasks in the OpenX training mixture is essential for unlocking improved generalization capabilities in the OpenVLA model.

<a id="table-9"></a>

> Table 9: BridgeData V2 WidowX ablation experiment results. We evaluate various methods on a subset of 8 representative tasks to assess the importance of different components of the OpenVLA model architecture and training scheme. OpenVLA-Bridge is a version of OpenVLA without OpenX training (it is trained only on BridgeData V2), and OpenVLA-Bridge-SigLIP additionally ablates the fused vision backbone by removing the DinoV2 encoder (its vision backbone only consists of the SigLIP encoder). We observe that both OpenX training and the fused vision encoder improve policy performance, though the former has a much greater effect than the latter.

| Category           | Task                                 | # Trials          | OpenVLA # Successes | OpenVLA-Bridge # Successes | OpenVLA-Bridge-SigLIP # Successes |
| ------------------ | ------------------------------------ | ----------------- | ------------------- | -------------------------- | --------------------------------- |
| Category           | Task                                 | # Trials          | OpenVLA # Successes | OpenVLA-Bridge # Successes | OpenVLA-Bridge-SigLIP # Successes |
| Category           | Task                                 | # Trials          | OpenVLA # Successes | OpenVLA-Bridge # Successes | OpenVLA-Bridge-SigLIP # Successes |
| Category           | Task                                 | # Trials          | OpenVLA # Successes | OpenVLA-Bridge # Successes | OpenVLA-Bridge-SigLIP # Successes |
| Category           | Task                                 | # Trials          | OpenVLA # Successes | OpenVLA-Bridge # Successes | OpenVLA-Bridge-SigLIP # Successes |
| Category           | Task                                 | # Trials          | OpenVLA # Successes | OpenVLA-Bridge # Successes | OpenVLA-Bridge-SigLIP # Successes |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                  | 8                          | 8                                 |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                  | 8                          | 8                                 |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                  | 8                          | 8                                 |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                  | 8                          | 8                                 |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                  | 8                          | 8                                 |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                  | 8                          | 8                                 |
| Visual gen         | Put Eggplant into Pot                | 10                | 10                  | 2                          | 3                                 |
| Visual gen         | Put Eggplant into Pot                | 10                | 10                  | 2                          | 3                                 |
| Visual gen         | Put Eggplant into Pot                | 10                | 10                  | 2                          | 3                                 |
| Visual gen         | Put Eggplant into Pot                | 10                | 10                  | 2                          | 3                                 |
| Visual gen         | Put Eggplant into Pot                | 10                | 10                  | 2                          | 3                                 |
| Visual gen         | Put Eggplant into Pot                | 10                | 10                  | 2                          | 3                                 |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 7                   | 4                          | 2                                 |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 7                   | 4                          | 2                                 |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 7                   | 4                          | 2                                 |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 7                   | 4                          | 2                                 |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 7                   | 4                          | 2                                 |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 7                   | 4                          | 2                                 |
| Motion gen         | Lift Eggplant                        | 10                | 7.5                 | 5.5                        | 6.5                               |
| Motion gen         | Lift Eggplant                        | 10                | 7.5                 | 5.5                        | 6.5                               |
| Motion gen         | Lift Eggplant                        | 10                | 7.5                 | 5.5                        | 6.5                               |
| Motion gen         | Lift Eggplant                        | 10                | 7.5                 | 5.5                        | 6.5                               |
| Motion gen         | Lift Eggplant                        | 10                | 7.5                 | 5.5                        | 6.5                               |
| Motion gen         | Lift Eggplant                        | 10                | 7.5                 | 5.5                        | 6.5                               |
| Physical gen       | Put Carrot on Plate                  | 10                | 8                   | 4                          | 1                                 |
| Physical gen       | Put Carrot on Plate                  | 10                | 8                   | 4                          | 1                                 |
| Physical gen       | Put Carrot on Plate                  | 10                | 8                   | 4                          | 1                                 |
| Physical gen       | Put Carrot on Plate                  | 10                | 8                   | 4                          | 1                                 |
| Physical gen       | Put Carrot on Plate                  | 10                | 8                   | 4                          | 1                                 |
| Physical gen       | Put Carrot on Plate                  | 10                | 8                   | 4                          | 1                                 |
| Physical gen       | Lift AAA Battery                     | 10                | 7                   | 2                          | 2                                 |
| Physical gen       | Lift AAA Battery                     | 10                | 7                   | 2                          | 2                                 |
| Physical gen       | Lift AAA Battery                     | 10                | 7                   | 2                          | 2                                 |
| Physical gen       | Lift AAA Battery                     | 10                | 7                   | 2                          | 2                                 |
| Physical gen       | Lift AAA Battery                     | 10                | 7                   | 2                          | 2                                 |
| Physical gen       | Lift AAA Battery                     | 10                | 7                   | 2                          | 2                                 |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 4                   | 3                          | 3                                 |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 4                   | 3                          | 3                                 |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 4                   | 3                          | 3                                 |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 4                   | 3                          | 3                                 |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 4                   | 3                          | 3                                 |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 4                   | 3                          | 3                                 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 7.5                 | 8                          | 7                                 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 7.5                 | 8                          | 7                                 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 7.5                 | 8                          | 7                                 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 7.5                 | 8                          | 7                                 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 7.5                 | 8                          | 7                                 |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 7.5                 | 8                          | 7                                 |
|                    |                                      | Mean Success Rate | 76.3 $\pm$ 4.8%     | 45.6 $\pm$ 5.6%            | 40.6 $\pm$ 5.5%                   |
|                    |                                      | Mean Success Rate | 76.3 $\pm$ 4.8%     | 45.6 $\pm$ 5.6%            | 40.6 $\pm$ 5.5%                   |
|                    |                                      | Mean Success Rate | 76.3 $\pm$ 4.8%     | 45.6 $\pm$ 5.6%            | 40.6 $\pm$ 5.5%                   |
|                    |                                      | Mean Success Rate | 76.3 $\pm$ 4.8%     | 45.6 $\pm$ 5.6%            | 40.6 $\pm$ 5.5%                   |
|                    |                                      | Mean Success Rate | 76.3 $\pm$ 4.8%     | 45.6 $\pm$ 5.6%            | 40.6 $\pm$ 5.5%                   |
|                    |                                      | Mean Success Rate | 76.3 $\pm$ 4.8%     | 45.6 $\pm$ 5.6%            | 40.6 $\pm$ 5.5%                   |

### D.2 Dual vs. Single Vision Encoder Experiments

The OpenVLA model architecture consists of a fused vision backbone that combines the SigLIP [[9]] and DinoV2 [[25]] encoders. In this section, we ablate the DinoV2 component to assess the importance of using a dual vision encoder.

Experimental setup and tasks. We instantiate a model, OpenVLA-Bridge-SigLIP, which is a version of OpenVLA that is trained only on BridgeData V2 and consists of only the SigLIP encoder as the vision backbone. We compare this model with the OpenVLA-Bridge model discussed in the previous section ([Section D.1](https://arxiv.org/html/2406.09246v3#A4.SS1)), which shares the same model architecture as the original OpenVLA model and is only trained on Bridge robot data. Therefore, the only difference between OpenVLA-Bridge-SigLIP and OpenVLA-Bridge is that the former omits the DinoV2 encoder in the vision backbone. We evaluate these models on the same subset of 8 Bridge tasks described in the previous section.

Results. Results for the dual vision encoder ablation are shown in [Table 9](#table-9). The drop in performance from OpenVLA-Bridge to OpenVLA-Bridge-SigLIP implies that additionally including the DinoV2 encoder in the vision backbone improves policy performance. However, the 5 percent reduction in performance here is not as significant as the 30 percent drop in performance observed from ablating OpenX training. The low-level spatial features represented in DinoV2 appear to aid generalization in only some cases.

### D.3 Fine-Tuned vs. Frozen Vision Encoder Experiments

As discussed in [Section 3.4](#section-3-4), prior work on VLMs observed higher performance from freezing the vision encoder than fine-tuning its parameters [[44]]. However, when training OpenVLA, we fine-tuned all 7B parameters in the model, including the SigLIP-DinoV2 vision backbone, as we discovered early on during development that fine-tuning the vision encoder led to higher-performing VLAs — a finding which held across various pretrained VLMs and model architectures. We discuss details of such findings below.

Experimental setup and tasks. In this section, we report the performance of two VLA policies produced by fine-tuning two different pretrained models from the Prismatic VLMs [[44]] repository on BridgeData V2. The two pretrained models are named SigLIP ViT-SO 224px and LLaVa v1.5 7B (Reproduction); see Karamcheti et al. [[44]] for details on their architectures and training mixtures. We evaluate both policies on various Bridge tasks shown in [Table 10](#table-10). Note that the evaluation configurations here differ from previously discussed Bridge evaluations, so the results are not directly comparable to results in other similar experiments.

Results. Results for the fine-tuned vs. frozen vision encoder experiments are shown in [Table 10](#table-10). We find that for both VLAs tested, fine-tuning the vision encoder leads to significantly higher success rates across various tasks. Qualitatively, in some cases, deploying the frozen vision encoder policies leads to unstable robot behaviors that are clearly suboptimal. Consequently, we decided early on during development to not conduct further experimentation with frozen vision encoders.

<a id="table-10"></a>

> Table 10: Fine-tuned vs. frozen vision encoder experiment results. We evaluate the performance of fine-tuning (“Fine-Tuned”) vs. freezing the vision encoder (“Frozen Vision”) in two VLA policies built on top of two different pretrained VLMs from the Prismatic VLMs [[44]] repository. BridgeData V2 WidowX tasks shown here are performed in the same sink environment used for other Bridge experiments in this work (however, the initial environment configurations here differ, as these evaluations were conducted at an earlier stage in the project). We find that fine-tuning the vision encoder is crucial to obtain good policy performance. Certain frozen vision encoder evaluations were discontinued due to very poor (near-zero) performance and unstable robot behaviors. Among the evaluations where both frozen vision and fine-tuned approaches are tested, fine-tuning the vision encoder leads to 80.0% average success versus 46.7% average success from leaving it frozen.

|                                     |                   | SigLIP ViT-SO 224px       | LLaVa v1.5 7B (Reproduction) |                           |                        |
| ----------------------------------- | ----------------- | ------------------------- | ---------------------------- | ------------------------- | ---------------------- |
|                                     |                   | SigLIP ViT-SO 224px       | LLaVa v1.5 7B (Reproduction) |                           |                        |
|                                     |                   | SigLIP ViT-SO 224px       | LLaVa v1.5 7B (Reproduction) |                           |                        |
|                                     |                   | SigLIP ViT-SO 224px       | LLaVa v1.5 7B (Reproduction) |                           |                        |
| Task                                | # Trials          | Frozen Vision # Successes | Fine-Tuned # Successes       | Frozen Vision # Successes | Fine-Tuned # Successes |
| Task                                | # Trials          | Frozen Vision # Successes | Fine-Tuned # Successes       | Frozen Vision # Successes | Fine-Tuned # Successes |
| Task                                | # Trials          | Frozen Vision # Successes | Fine-Tuned # Successes       | Frozen Vision # Successes | Fine-Tuned # Successes |
| Task                                | # Trials          | Frozen Vision # Successes | Fine-Tuned # Successes       | Frozen Vision # Successes | Fine-Tuned # Successes |
| Task                                | # Trials          | Frozen Vision # Successes | Fine-Tuned # Successes       | Frozen Vision # Successes | Fine-Tuned # Successes |
| Task                                | # Trials          | Frozen Vision # Successes | Fine-Tuned # Successes       | Frozen Vision # Successes | Fine-Tuned # Successes |
| Put Eggplant into Pot               | 10                | 7                         | 10                           | 5                         | 9                      |
| Put Eggplant into Pot               | 10                | 7                         | 10                           | 5                         | 9                      |
| Put Eggplant into Pot               | 10                | 7                         | 10                           | 5                         | 9                      |
| Put Eggplant into Pot               | 10                | 7                         | 10                           | 5                         | 9                      |
| Put Eggplant into Pot               | 10                | 7                         | 10                           | 5                         | 9                      |
| Put Eggplant into Pot               | 10                | 7                         | 10                           | 5                         | 9                      |
| Put Corn on Plate                   | 10                | 10                        | 9                            | 0                         | 9                      |
| Put Corn on Plate                   | 10                | 10                        | 9                            | 0                         | 9                      |
| Put Corn on Plate                   | 10                | 10                        | 9                            | 0                         | 9                      |
| Put Corn on Plate                   | 10                | 10                        | 9                            | 0                         | 9                      |
| Put Corn on Plate                   | 10                | 10                        | 9                            | 0                         | 9                      |
| Put Corn on Plate                   | 10                | 10                        | 9                            | 0                         | 9                      |
|                                     | Mean Success Rate | 85                        | 95                           | 25                        | 90                     |
|                                     | Mean Success Rate | 85                        | 95                           | 25                        | 90                     |
|                                     | Mean Success Rate | 85                        | 95                           | 25                        | 90                     |
|                                     | Mean Success Rate | 85                        | 95                           | 25                        | 90                     |
|                                     | Mean Success Rate | 85                        | 95                           | 25                        | 90                     |
|                                     | Mean Success Rate | 85                        | 95                           | 25                        | 90                     |
| Put {Eggplant, Red Bottle} into Pot | 4                 | 2                         | 4                            | –                         | 3                      |
| Put {Eggplant, Red Bottle} into Pot | 4                 | 2                         | 4                            | –                         | 3                      |
| Put {Eggplant, Red Bottle} into Pot | 4                 | 2                         | 4                            | –                         | 3                      |
| Put {Eggplant, Red Bottle} into Pot | 4                 | 2                         | 4                            | –                         | 3                      |
| Put {Eggplant, Red Bottle} into Pot | 4                 | 2                         | 4                            | –                         | 3                      |
| Put {Eggplant, Red Bottle} into Pot | 4                 | 2                         | 4                            | –                         | 3                      |
| Put {Blue Cup, Pink Cup} on Plate   | 4                 | 0                         | 0                            | –                         | 0                      |
| Put {Blue Cup, Pink Cup} on Plate   | 4                 | 0                         | 0                            | –                         | 0                      |
| Put {Blue Cup, Pink Cup} on Plate   | 4                 | 0                         | 0                            | –                         | 0                      |
| Put {Blue Cup, Pink Cup} on Plate   | 4                 | 0                         | 0                            | –                         | 0                      |
| Put {Blue Cup, Pink Cup} on Plate   | 4                 | 0                         | 0                            | –                         | 0                      |
| Put {Blue Cup, Pink Cup} on Plate   | 4                 | 0                         | 0                            | –                         | 0                      |
| Lift {Cheese, Red Chili Pepper}     | 4                 | 0                         | 3                            | –                         | 2                      |
| Lift {Cheese, Red Chili Pepper}     | 4                 | 0                         | 3                            | –                         | 2                      |
| Lift {Cheese, Red Chili Pepper}     | 4                 | 0                         | 3                            | –                         | 2                      |
| Lift {Cheese, Red Chili Pepper}     | 4                 | 0                         | 3                            | –                         | 2                      |
| Lift {Cheese, Red Chili Pepper}     | 4                 | 0                         | 3                            | –                         | 2                      |
| Lift {Cheese, Red Chili Pepper}     | 4                 | 0                         | 3                            | –                         | 2                      |
| Put {Strawberry, Lime} into Pot     | 4                 | 1                         | 0                            | –                         | 3                      |
| Put {Strawberry, Lime} into Pot     | 4                 | 1                         | 0                            | –                         | 3                      |
| Put {Strawberry, Lime} into Pot     | 4                 | 1                         | 0                            | –                         | 3                      |
| Put {Strawberry, Lime} into Pot     | 4                 | 1                         | 0                            | –                         | 3                      |
| Put {Strawberry, Lime} into Pot     | 4                 | 1                         | 0                            | –                         | 3                      |
| Put {Strawberry, Lime} into Pot     | 4                 | 1                         | 0                            | –                         | 3                      |
| Move {Sushi, Grapes}                | 4                 | 3                         | 4                            | –                         | 3                      |
| Move {Sushi, Grapes}                | 4                 | 3                         | 4                            | –                         | 3                      |
| Move {Sushi, Grapes}                | 4                 | 3                         | 4                            | –                         | 3                      |
| Move {Sushi, Grapes}                | 4                 | 3                         | 4                            | –                         | 3                      |
| Move {Sushi, Grapes}                | 4                 | 3                         | 4                            | –                         | 3                      |
| Move {Sushi, Grapes}                | 4                 | 3                         | 4                            | –                         | 3                      |
|                                     | Mean Success Rate | 30                        | 55                           | –                         | 55                     |
|                                     | Mean Success Rate | 30                        | 55                           | –                         | 55                     |
|                                     | Mean Success Rate | 30                        | 55                           | –                         | 55                     |
|                                     | Mean Success Rate | 30                        | 55                           | –                         | 55                     |
|                                     | Mean Success Rate | 30                        | 55                           | –                         | 55                     |
|                                     | Mean Success Rate | 30                        | 55                           | –                         | 55                     |

### D.4 Additional Quantized Inference Experiments: Disentangling Policy Performance and Model Inference Speed

In [Section 5.3](#section-5-3), we evaluated OpenVLA with different levels of precision at inference time: half precision (bfloat16), 8-bit quantization, and 4-bit quantization. 8-bit quantization led to lower BridgeData V2 performance relative to the other two approaches, and we hypothesized that the reduction in performance was caused by lower model inference speed from the operations used in 8-bit quantization. In this section, we conduct experiments to assess the veracity of this claim.

Specifically, we evaluate OpenVLA again with the three different levels of precision listed above, but now with _blocking control_. In other words, each action is fully executed on the robot before the next one is predicted by the policy and executed by the controller. This scheme controls system dynamics across methods with varying amounts of latency and thus allows us to test the quality of a policy’s action predictions, independent of its prediction speed. Effectively, the precision levels that have higher throughput – bfloat16 and 4-bit quantization – are forced to run slower to match the dynamics observed when deploying OpenVLA with 8-bit precision. Therefore, we expect OpenVLA’s performance with 8-bit precision to match the performance of bfloat16 and 4-bit precision under blocking control.

Experimental setup and tasks. We report the performance of OpenVLA with blocking control and quantized inference on the same subset of 8 BridgeData V2 tasks used in [Section D.1](https://arxiv.org/html/2406.09246v3#A4.SS1) and [Section D.2](https://arxiv.org/html/2406.09246v3#A4.SS2).

Results. Quantized inference experiment results with blocking control are shown in [Table 11](#table-11). Unlike in [Table 2](#table-2), where 8-bit quantization led to the worst rollout performance due to low inference speed, here we observe that 8-bit quantization performs comparably to bfloat16 precision and 4-bit quantization given that we evaluate with blocking control to remove the influence of varying inference speeds on task performance. This confirms our hypothesis about the effect of inference speed on 8-bit quantization performance in previous experiments (when using non-blocking control). We also see no substantial performance degradation when using the lowest precision, 4-bit, as also observed in [Section 5.3](#section-5-3).

<a id="table-11"></a>

> Table 11: Quantized inference experiment results with blocking control. We report the success rate and standard error of OpenVLA on various BridgeData V2 WidowX tasks with bfloat16 precision (the default approach), 8-bit quantization (int8), and 4-bit quantization (int4) at inference time. All average success rates have overlapping error bars, which suggests that all methods perform comparably.

| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| ------------------ | ------------------------------------ | ----------------- | -------------------- | ---------------- | ---------------- |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Category           | Task                                 | # Trials          | bfloat16 # Successes | int8 # Successes | int4 # Successes |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                   | 10               | 10               |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                   | 10               | 10               |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                   | 10               | 10               |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                   | 10               | 10               |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                   | 10               | 10               |
| Visual gen         | Put Eggplant into Pot (Easy Version) | 10                | 10                   | 10               | 10               |
| Visual gen         | Put Eggplant into Pot                | 10                | 9                    | 10               | 10               |
| Visual gen         | Put Eggplant into Pot                | 10                | 9                    | 10               | 10               |
| Visual gen         | Put Eggplant into Pot                | 10                | 9                    | 10               | 10               |
| Visual gen         | Put Eggplant into Pot                | 10                | 9                    | 10               | 10               |
| Visual gen         | Put Eggplant into Pot                | 10                | 9                    | 10               | 10               |
| Visual gen         | Put Eggplant into Pot                | 10                | 9                    | 10               | 10               |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 5                | 3                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 5                | 3                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 5                | 3                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 5                | 3                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 5                | 3                |
| Visual gen         | Put Cup from Counter into Sink       | 10                | 5                    | 5                | 3                |
| Motion gen         | Lift Eggplant                        | 10                | 8                    | 7                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 8                    | 7                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 8                    | 7                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 8                    | 7                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 8                    | 7                | 7.5              |
| Motion gen         | Lift Eggplant                        | 10                | 8                    | 7                | 7.5              |
| Physical gen       | Put Carrot on Plate                  | 10                | 10                   | 10               | 10               |
| Physical gen       | Put Carrot on Plate                  | 10                | 10                   | 10               | 10               |
| Physical gen       | Put Carrot on Plate                  | 10                | 10                   | 10               | 10               |
| Physical gen       | Put Carrot on Plate                  | 10                | 10                   | 10               | 10               |
| Physical gen       | Put Carrot on Plate                  | 10                | 10                   | 10               | 10               |
| Physical gen       | Put Carrot on Plate                  | 10                | 10                   | 10               | 10               |
| Physical gen       | Lift AAA Battery                     | 10                | 3                    | 6                | 4                |
| Physical gen       | Lift AAA Battery                     | 10                | 3                    | 6                | 4                |
| Physical gen       | Lift AAA Battery                     | 10                | 3                    | 6                | 4                |
| Physical gen       | Lift AAA Battery                     | 10                | 3                    | 6                | 4                |
| Physical gen       | Lift AAA Battery                     | 10                | 3                    | 6                | 4                |
| Physical gen       | Lift AAA Battery                     | 10                | 3                    | 6                | 4                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 2                    | 2                | 2                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 2                    | 2                | 2                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 2                    | 2                | 2                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 2                    | 2                | 2                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 2                    | 2                | 2                |
| Semantic gen       | Take Purple Grapes out of Pot        | 10                | 2                    | 2                | 2                |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 9.5              | 8.5              |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 9.5              | 8.5              |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 9.5              | 8.5              |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 9.5              | 8.5              |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 9.5              | 8.5              |
| Language grounding | Put {Eggplant, Red Bottle} into Pot  | 10                | 9                    | 9.5              | 8.5              |
|                    |                                      | Mean Success Rate | 70.0 $\pm$ 5.1%      | 74.4 $\pm$ 4.9%  | 68.8 $\pm$ 5.2%  |
|                    |                                      | Mean Success Rate | 70.0 $\pm$ 5.1%      | 74.4 $\pm$ 4.9%  | 68.8 $\pm$ 5.2%  |
|                    |                                      | Mean Success Rate | 70.0 $\pm$ 5.1%      | 74.4 $\pm$ 4.9%  | 68.8 $\pm$ 5.2%  |
|                    |                                      | Mean Success Rate | 70.0 $\pm$ 5.1%      | 74.4 $\pm$ 4.9%  | 68.8 $\pm$ 5.2%  |
|                    |                                      | Mean Success Rate | 70.0 $\pm$ 5.1%      | 74.4 $\pm$ 4.9%  | 68.8 $\pm$ 5.2%  |
|                    |                                      | Mean Success Rate | 70.0 $\pm$ 5.1%      | 74.4 $\pm$ 4.9%  | 68.8 $\pm$ 5.2%  |

<a id="appendix-e"></a>

## Appendix E LIBERO Simulation Experiments

Our previous discussions in [Section 5.2](#section-5-2) and [Section 5.3](#section-5-3) focused on adapting OpenVLA to novel _real-world_ robot setups and tasks. This section explores adapting OpenVLA to _simulated_ robot setups and tasks, specifically utilizing the LIBERO benchmark [[116]]. Our experimentation in simulation offers two key advantages:

- 1. Demonstration of versatility: We show that OpenVLA, despite having been pretrained exclusively on real-world robot data, can effectively adapt to simulated domains, overcoming potential disparities between real-world and simulated environments and dynamics.
- 2. Enhanced accessibility and reproducibility: Integration of OpenVLA into a publicly available simulation platform makes our model more accessible to other researchers, especially those who may not have access to robotic hardware. Additionally, simulated experiments are more easily reproduced than their real-world counterparts.

We discuss the experimental setup in [Section E.1](https://arxiv.org/html/2406.09246v3#A5.SS1) and the results in [Section E.2](https://arxiv.org/html/2406.09246v3#A5.SS2). We release the materials required to reproduce the experiments along with the OpenVLA codebase.

- [E.1 LIBERO Simulation Experimental Setup](#e1-libero-simulation-experimental-setup)
- [E.2 LIBERO Simulation Experimental Results](#e2-libero-simulation-experimental-results)

### E.1 LIBERO Simulation Experimental Setup

Simulation setup and tasks. The LIBERO benchmark [[116]] consists of four task suites designed for studying lifelong learning in robotic manipulation, and the original paper therefore investigates both forward and backward transfer to a variety of tasks. In our experiments, we focus solely on supervised fine-tuning on the target task suite, measuring the performance of various policies trained via behavioral cloning on successful demonstrations of the tasks.

We perform experiments with the following four task suites, which each contain 10 tasks with 50 human-teleoperated demonstrations each:

- LIBERO-Spatial consists of the same set of objects but different layouts, and tests the model’s understanding of spatial relationships.
- LIBERO-Object consists of the same scene layouts but different objects, and tests the model’s understanding of object types.
- LIBERO-Goal consists of the same objects and layouts but different task goals, and tests the model’s knowledge of different task-oriented behaviors.
- LIBERO-Long (also called LIBERO-10) consists of _long-horizon_ tasks with diverse objects, layouts, and tasks.

We make the following modifications to each of the training datasets above:

- 1. To accommodate methods requiring higher-resolution images (such as $256\times 256$px or $224\times 224$px), we regenerate all demonstrations at an increased resolution of $256\times 256$px. Originally, the dataset provided by the benchmark consists of $128\times 128$px images. We find that simply upscaling these images to $256\times 256$px results in poor image quality. Therefore, we choose to begin with higher-resolution images, which can be downscaled as necessary, ensuring higher image quality across various resolution requirements. These higher-resolution images were obtained by stepping through the simulation environments with the actions stored in the provided human-collected demonstrations and saving the images rendered by the simulator.
- 2. We filter out all “no-op” actions from the dataset, i.e., actions that have near-zero magnitude in the translation and rotation components and do not change the state of the robot’s gripper. We find that this simple data cleaning step is crucial for highly expressive single-step policies such as OpenVLA, which otherwise learn to imitate these no-op actions and consequently freeze indefinitely at certain states during evaluation.
- 3. We rotate all third-person images at both train and test time by 180 degrees because we observe that the LIBERO environments return images that are upside down on our hardware.
- 4. Since we train policies via imitation learning, which expects demonstrations to be successful, we replay all demonstrations in the corresponding simulation environments and filter out the demonstrations that fail to complete the task (as determined by the environments’ success criteria). As a result, we remove 68 of 500 LIBERO-Spatial demonstrations, 46 of 500 LIBERO-Object demonstrations, 72 of 500 LIBERO-Goal demonstrations, and 121 of 500 LIBERO-Long demonstratinos.
- 5. For all methods in our comparisons, we only utilize the static third-person camera images; we do not use the wrist camera images that are additionally provided in the original datasets. This is for sake of having fair comparisons, as OpenVLA’s visual inputs only consist of third-person camera images.

Comparisons. The methods that we compare include Diffusion Policy(^8^88We use the implementation of Diffusion Policy that is described in the DROID dataset paper [[11]], which conditions action generation on DistilBERT [[117]] language embeddings of the task label.) [[3]] trained from scratch, Octo [[5]] fine-tuned on the target dataset, and OpenVLA fine-tuned on the target dataset via LoRA ($r=32$) as described in [Section 5.3](#section-5-3). Each policy is trained independently on each of the task suites above (rather than training a single policy on all four suites combined). All policies are trained with the same set of demonstrations, so all methods benefit from the data cleaning steps described above.

Evaluation details. To ensure lower variance in the experimental results, all methods are evaluated across 500 trials for each task suite, and the reported performance is the average success rate over three random seeds (resulting in 1500 total trials per statistic). Although we modify the training datasets, as described earlier, we do not change the test environments but rather use the same initial environment configurations provided by the original LIBERO benchmark.

### E.2 LIBERO Simulation Experimental Results

We present the LIBERO experimental results in [Table 12](#table-12). Importantly, we observe that OpenVLA can be effectively adapted to tasks in the LIBERO simulation environments, as it obtains highest average success rate and rank among the tested methods. However, we find that the overall margin between OpenVLA and the other methods are tighter here than in the real-world fine-tuning experiments discussed in [Section 5.2](#section-5-2). We attribute this to the fact that OpenVLA was pretrained with purely real-world robot data and no simulation data, which suggests that fine-tuning the model on simulated robot tasks may not be as effective as fine-tuning it on real-world tasks due to the domain gap between simulated and real-world environments and dynamics. We see evidence for this notion in the results obtained by Octo – another policy pretrained on large amounts of real-world robot data – which also only achieves a small boost in overall performance relative to a simple, strong baseline such as Diffusion Policy trained from scratch. We expect increased gains in performance for the pretrained and fine-tuned methods if simulation data is added to the pretraining data mixture.

<a id="table-12"></a>

> Table 12: LIBERO simulation benchmark results. We report the success rate (SR) and standard error of each method for the four task suites in the LIBERO benchmark, averaged over three random seeds with 500 trials each. In addition, we show the ranking of each method within each task suite, where a rank of 1 indicates the strongest method in the suite and a rank of 3 indicates the weakest method. (The average ranking is important to note since it informs which method may be most suitable to use as a default for a variety of tasks; it is more informative than the average success rate, which is not normalized by individual task suite difficulty.) Overall, we find that fine-tuned OpenVLA achieves highest average success rate and rank, followed by fine-tuned Octo and then Diffusion Policy trained from scratch.

|                               | LIBERO-Spatial  | LIBERO-Object       | LIBERO-Goal     | LIBERO-Long         | Average         |                     |                 |                     |                 |                     |
| ----------------------------- | --------------- | ------------------- | --------------- | ------------------- | --------------- | ------------------- | --------------- | ------------------- | --------------- | ------------------- |
|                               | LIBERO-Spatial  | LIBERO-Object       | LIBERO-Goal     | LIBERO-Long         | Average         |                     |                 |                     |                 |                     |
|                               | LIBERO-Spatial  | LIBERO-Object       | LIBERO-Goal     | LIBERO-Long         | Average         |                     |                 |                     |                 |                     |
|                               | LIBERO-Spatial  | LIBERO-Object       | LIBERO-Goal     | LIBERO-Long         | Average         |                     |                 |                     |                 |                     |
|                               | LIBERO-Spatial  | LIBERO-Object       | LIBERO-Goal     | LIBERO-Long         | Average         |                     |                 |                     |                 |                     |
|                               | LIBERO-Spatial  | LIBERO-Object       | LIBERO-Goal     | LIBERO-Long         | Average         |                     |                 |                     |                 |                     |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
|                               | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) | SR ($\uparrow$) | Rank ($\downarrow$) |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Diffusion Policy from scratch | 78.3 $\pm$ 1.1% | 3                   | 92.5 $\pm$ 0.7% | 1                   | 68.3 $\pm$ 1.2% | 3                   | 50.5 $\pm$ 1.3% | 3                   | 72.4 $\pm$ 0.7% | 2.5                 |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| Octo fine-tuned               | 78.9 $\pm$ 1.0% | 2                   | 85.7 $\pm$ 0.9% | 3                   | 84.6 $\pm$ 0.9% | 1                   | 51.1 $\pm$ 1.3% | 2                   | 75.1 $\pm$ 0.6% | 2                   |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
| OpenVLA fine-tuned (ours)     | 84.7 $\pm$ 0.9% | 1                   | 88.4 $\pm$ 0.8% | 2                   | 79.2 $\pm$ 1.0% | 2                   | 53.7 $\pm$ 1.3% | 1                   | 76.5 $\pm$ 0.6% | 1.5                 |
