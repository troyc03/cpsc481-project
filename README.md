# Discovering Chaotic Dynamics with Mathematical Physics: An AI-Assisted Analysis of the Lorenz System using Machine Learning

### **Description**: 
This project applies artificial intelligence methods to discover and model chaotic systems. Using the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm, we train a model to learn the governing equations of the Lorenz attractor directly from simulated data. This approach embodies a key idea in AI — learning structure from data — while emphasizing interpretability and scientific reasoning. The project combines applied mathematics, machine learning, and data visualization through an interactive GUI that allows users to generate Lorenz system data, apply SINDy-based model discovery, and visualize the reconstructed trajectories and learned equations. This project highlights interpretable AI applied to nonlinear systems and demonstrates how sparse regression techniques can uncover governing dynamics from observed data.

### **Group Members**: Troy Chin

**System Requirements**:
Windows is the recommended OS for running this program, as well as the latest Python version (Python 3.13.7). Python 3.11 and 3.12 are also acceptable.

### **Installation**:
**Option 1**: In either VSCode or your Windows Powershell console, please install the following libraries:

```bash
pip install numpy matplotlib scipy scikit-learn pysindy pandas
```
**Option 2**: For non-VSCode users, install the Spyder IDE as each of these libraries come pre-installed. You may still have to install PySINDy and ScikitLearn manually.
[Spyder IDE/]([url](https://www.spyder-ide.org/)) 

### **Running the Application**:
To run the main simulation, enter the following prompt:

```
python -m lorenz_attractor
python -m particle_kinematics
python -m lorenz_lyapunov_exp
```
To run the model analysis program, enter the following prompt:

```
python -m evaluate
```

### **Algorithms**: 
1. Sparse Identification of Nonlinear Dynamics (SINDy) 
2. Ordinary Differential Equation Solvers (Runge–Kutta methods via SciPy)
3. Stochastic Gradient Descent
4. Linear Regression
5. Root Mean Squared Error

### **Libraries**: The Lorenz attractor equations cannot be solved analytically - in order to solve them numerically, this project will be using the following:
- NumPy for numerical computing
- SciPy for solving differential equations numerically
- Matplotlib for graphing the Lorenz attractor/visualization
- Pandas for data preprocessing
- CustomTkinter for GUI development
- Scikit-Learn for Evaluation and Analysis
- PySINDy for deriving and solving nonlinear dynamical systems

### **Timeline**: This project will be split into three phases:
- Phase I: Background Research (Week 1 - 2)- Intensive research on Sparse Identification of Nonlinear Dynamics theory and nonlinear dynamics; generation and implementation of Lorenz attractor data.
- Phase II: Mathematical Modeling/Foundations (Week 4 - 6) - Derive the governing equations of the Lorenz attractor for computational implementation; all derivations will employ the following concepts - ordinary differential equations, nonlinear dynamical systems and chaos theory. 
- Phase III: Computational Implementation (Week 7 - 8) - Prepare for computational implementation of the Lorenz equations into Python (and MATLAB). This part of the project is a culmination of phases 1 and 2. Once everything is finished, a report will be written in LaTeX to simulate and describe the dynamics of the Lorenz attractor.

### **Roles and Responsibilities**: -
- Troy Chin: Project lead, Lorenz system implementation and mathematical implementations, report writing. 
