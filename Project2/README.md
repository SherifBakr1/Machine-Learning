[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ph4wpRU3)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-7f7980b617ed060a017424585567c406b6ee15c891e84e1186181d67ecf80aa0.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=11954526)
# Introduction
In this homework, your Pacman agent will find paths through his maze world, both to reach a particular location and to collect food efficiently. 

The starter repository contains the Pacman game around which you will be developing code to implement various search techniques. You should unzip the file and run it to get an idea of how it works. For instance, to play a game of classic Pacman, run:

```python pacman.py```

Pacman lives in a shiny blue world of twisting corridors and tasty round treats. Navigating this world efficiently will be Pacman's first step in mastering his domain.

**Note:** A description of the files involved in this project can be found in Appendix [Files](#Files) at the end of this document. 

**This assignment is to be completed in Python 3**

## Getting Help
We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask. If you find yourself stuck on something, contact us via Slack or come by the office hours. If you can't make our office hours, let us know and we will be happy to schedule alternate times.

# Agents
The simplest agent in `searchAgents.py` is called the GoWestAgent, which always goes West (a trivial reflex agent). This agent can occasionally win:

```python pacman.py --layout testMaze --pacman GoWestAgent```

But, things get ugly for this agent when turning is required:

```python pacman.py --layout tinyMaze --pacman GoWestAgent```

If Pacman gets stuck, you can exit the game by typing `CTRL+c` (or equivalent) into your terminal.

Soon, your agent will solve not only tinyMaze, but any maze you want.

Note that `pacman.py` supports a number of options that can each be expressed in a long way (e.g., `--layout`) or a short way (e.g., `-l`). You can see the list of all options and their default values via:
    
```python pacman.py -h```

Also, all of the commands that appear in this project also appear in `commands.txt`, for easy copying and pasting. In UNIX/Mac OS X, you can even run all these commands in order with bash `commands.txt`.

# Useful Information

**Checking Your Work**, you can check your code by running 
    
```python autograder.py```

This will run your functions through a series of test cases, similar to those used on gradescope. Questions 4b and 5 are not autograded, so the autograder will only give you a score out of 18, the remaining 7 points will be manually graded. 

**Files to Edit and Submit:** You will fill in portions of `search.py` and `serchAgents.py`. You should submit these files with your code and comments, as well as your final report. Please do not change the other files in this distribution or submit any of our original files other than these files. You may want to review other files in the project, these have been shown in Appendix [Files](#Files).

**Evaluation:** Your code will be autograded for technical correctness. Please **do not** change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation -- not the autograder's judgements -- will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work.

**For Full Question Deatails** see assignment details on Canvas


# File List
---

**Files you'll edit:**

| **File**                | **Description**                                                                                                                  |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| search.py               | Where all of your search algorithms will reside.                                                                                  |
| searchAgents.py         | Where all of your search-based agents will reside.                                                                                |

**Files you might want to look at:**

| **File**                | **Description**                                                                                                                  |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| pacman.py               | The main file that runs Pacman games. This file describes a Pacman GameState type, which you use in this project.                 |
| game.py                 | The logic behind how the Pacman world works. This file describes several supporting types like AgentState, Agent, Direction, and Grid. |
| util.py                 | Useful data structures for implementing search algorithms.                                                                        |

**Supporting files you can ignore:**

| **File**                | **Description**                                                                                                                  |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| graphicsDisplay.py      | Graphics for Pacman                                                                                                               |
| graphicsUtils.py        | Support for Pacman graphics                                                                                                       |
| textDisplay.py          | ASCII graphics for Pacman                                                                                                         |
| ghostAgents.py          | Agents to control ghosts                                                                                                          |
| keyboardAgents.py      | Keyboard interfaces to control Pacman                                                                                             |
| layout.py               | Code for reading layout files and storing their contents                                                                          |
| autograder.py           | Project autograder                                                                                                                |
| testParser.py           | Parses autograder test and solution files                                                                                         |
| testClasses.py          | General autograding test classes                                                                                                  |
| test_cases/             | Directory containing the test cases for each question                                                                             |
| searchTestClasses.py    | Project 1 specific autograding test classes                                                                                       |

---


