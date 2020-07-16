# encoding: utf-8

# default parameters
readParamsFromTable(
    # Density
    rho=2450,
    # Young's modulus
    E=70e+9,
    # Poisson's ratio
    nu=0.2,
    # final friction coefficient
    mu=0.4,
    # timestepSafetyCoefficient
    safe=0.1,
    # no. of your simulation
    key=0
)

import numpy as np
from yade.params import table
from yade import plot

# glass bead parameters (units: ug->1e-9kg; mm->1e-3m; ms->1e-3s)
lenScale = 1e3  # lenth in mm <- 1e-3 m
sigScale = 1  # Stress in ug/(mm*ms^2) <- Pa
rhoScale = 1  # Density in ug/mm^3 <- kg/m^3


# function to save simulation data and stop simulation
def addSimData():
    inter = O.interactions[0, 1]
    u = inter.geom.penetrationDepth
    if u > obsCtrlData[-1]:
        plot.addData(u=u, f=inter.phys.normalForce.norm())
        obsCtrlData.pop()
    if not obsCtrlData: O.pause()


# get data for simulation control
obsCtrlData = list(np.loadtxt('collision.dat')[0, :])
obsCtrlData.reverse()

print('\nModel evaluation NO. %i' % table.key)

# create materials
O.materials.append(FrictMat(young=table.E, poisson=table.nu, frictionAngle=atan(table.mu), density=table.rho))

# create two particles
O.bodies.append(sphere(Vector3(0, 0, 0), 1, material=0, fixed=True))
O.bodies.append(sphere(Vector3(0, 0, 2), 1, material=0, fixed=True))

# define engines
O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_MindlinPhys()],
        [Law2_ScGeom_MindlinPhys_Mindlin()]
    ),
    NewtonIntegrator(damping=0.0, label='newton'),
    # needs to add module collision before function name
    PyRunner(command='addSimData()', iterPeriod=100)
]

# set initial timestep
O.dt = table.safe * PWaveTimeStep()
# move particle 1
O.bodies[1].state.vel = Vector3(0, 0, -0.01)

# plotting inside Yade
plot.plots = {'u': 'f'}

# run DEM simulation
O.run(int(1e10))
waitIfBatch()