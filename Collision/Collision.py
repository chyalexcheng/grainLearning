# encoding: utf-8

# default parameters
readParamsFromTable(
    # Density
    rho=2450,
    # Young's modulus
    E_m=8,
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

isBatch = runningInBatch()	# check if run in batch mode
if isBatch:
	print('Running: '+O.tags['description'])

# glass bead parameters (units: ug->1e-9kg; mm->1e-3m; ms->1e-3s)
lenScale = 1e3  # length in mm <- 1e-3 m
sigScale = 1  # Stress in ug/(mm*ms^2) <- Pa
rhoScale = 1  # Density in ug/mm^3 <- kg/m^3


# function to save simulation data and stop simulation
def addSimData():
    inter = O.interactions[0, 1]
    u = inter.geom.penetrationDepth
    if u > obsCtrlData[-1]:
        plot.addData(u=u, f=inter.phys.normalForce.norm())
        obsCtrlData.pop()
    if not obsCtrlData:
        if isBatch:
			# store simulation data and parameter set as a dictory in the .npy file
            dataName = '2particle_'+ O.tags['description'] + '.npy'
            for name in table.__all__: plot.data[name] = eval('table.'+name)
            np.save(dataName,plot.data)
        else:
			# File name: <simName>_<key>_<param0>_<param1>_..._<paramN>.txt
            dataName = '2particle_%i_%.10e_%.10e' % (table.key, table.E, table.nu) + '.txt'
            plot.saveDataTxt(dataName)
        O.pause()

obsFile ="collisionOrg.dat"
# get data for simulation control
obsCtrlData = np.loadtxt(obsFile)[:,0].tolist()
obsCtrlData.reverse()

print('\nModel evaluation NO. %i' % table.key)

# create materials
O.materials.append(FrictMat(young=pow(10,table.E_m), poisson=table.nu, frictionAngle=atan(table.mu), density=table.rho))

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
    PyRunner(command='addSimData()', iterPeriod=1)
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
