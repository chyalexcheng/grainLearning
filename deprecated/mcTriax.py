# encoding: utf-8
""" Quasi-static drained triaxial compression on Toyoura sand
    (e_0 = 0.68) under 0.5 MPa confining pressure. Simulation data 
    are extracted at every 0.1% axial strain increment until axial
    compression reaches 10%.
"""

# read parameter values from table
readParamsFromTable(
   E = 4e9,
   v = 0.33,
   kr = 0.13,
   eta = 0.13,
   mu = 29.0,
   num = 1000,
   conf = 0.5e6,
   key = 1
)

from yade.params import table
from yade import pack, plot

# Strain goal and number of simulation steps (same as test data)
strainGoal = 0.1 	        # target strain level
nstep = 100               # number of simulation steps
dstrain = strainGoal/nstep# strain increment

# Simulation control
random = False            # use ramdom or pre-defined particle packing
num = table.num           # number of soil particles
densityScaling = 1e3      # density scaling
e = 0.68                  # initial void ratio
conf = table.conf         # confining pressure
rate = 0.1                # strain rate
damp = 0.2                # damping coefficient
stressTolRatio = 1.e-3    # stress tolerance ratio
stabilityRatio = 1.e-3    # stability threshold

# Soil sphere parameters
E=table.E                 # micro Young's modulus
v=table.v                 # micro Poisson's ratio
kr=table.kr               # rolling/bending stiffness
eta=table.eta             # rolling/bending plastic limit
mu = table.mu             # contact friction during shear
rho = 2650*densityScaling # soil density

# create materials
spMat = O.materials.append(
   CohFrictMat(young=E,poisson=v,frictionAngle=radians(mu),density=rho,isCohesive=False,
      alphaKr=kr,alphaKtw=kr,momentRotationLaw=True,etaRoll=kr,etaTwist=kr))
	
# create a cloud of ramdomly positioned spheres
O.periodic = True
sp=pack.SpherePack()

# if use random packing
if random:
	# corners to define specimen size
   mn,mx=Vector3.Zero,Vector3(0.1,0.1,0.2)
   O.cell.hSize = Matrix3(mx[0],0,0, 0,mx[1],0, 0,0,mx[2])
   sizes=[.00575,.00685,.00816,.00969,.01150,.01369,.01626]
   cumm=[.013,.021,.058,.174,.811,.927,1]
   sp.makeCloud(minCorner=mn,maxCorner=mx,psdSizes=sizes,psdCumm=cumm,\
      distributeMass=True,porosity=e/(1+e),seed=1,num=num)
   # reduce fricion angle to densify random packing
   ctrMu = 0.5*table.mu
   O.materials[spMat].frictionAngle = radians(ctrMu)
# if use a pre-defined packing
else:
   if num==1000: O.cell.hSize=Matrix3(0.04622,0,0, 0,0.04612,0, 0,0,0.09212)
   if num==2000: O.cell.hSize=Matrix3(0.05013,0,0, 0,0.05015,0, 0,0,0.09945)
   if num==5000: O.cell.hSize=Matrix3(0.04617,0,0, 0,0.04626,0, 0,0,0.09201)
   if num==10000:O.cell.hSize=Matrix3(0.04609,0,0, 0,0.04616,0, 0,0,0.09221)
   sp.load('../PeriSp_'+str(num)+'_'+str(e)+'.txt')

# load spheres to simulation
spIds=sp.toSimulation(material=spMat)

# define engines
O.engines=[
   ForceResetter(),
   InsertionSortCollider([Bo1_Sphere_Aabb()]),
   InteractionLoop(
		[Ig2_Sphere_Sphere_ScGeom6D()],
		[Ip2_CohFrictMat_CohFrictMat_CohFrictPhys()],
		[Law2_ScGeom6D_CohFrictPhys_CohesionMoment(
		   always_use_moment_law=True,
		   useIncrementalForm=True
		)],
   ),
   GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.8),
   PeriTriaxController(label='triax',
      # whether they are strains or stresses
      stressMask=7,
      # type of servo-control
      dynCell=True,
      # wait until the unbalanced force goes below this value
      maxUnbalanced=stabilityRatio,
      # turn on checkVoidRatio after finishing initial compression
      relStressTol=stressTolRatio,
   ),
   NewtonIntegrator(damping=damp,label='newton'),
   ]

# prepare random dense particle packing
if random:
   triaxDone = False
   triax.goal=(-.01*conf,-0.01*conf,-0.01*conf)
   #~ triax.goal=(-100,-100,-100)
   triax.maxStrainRate=(10.*rate,10.*rate,10.*rate)
   triax.doneHook="triaxDone=True; newton.damping=0.9"
   # prepare dense packing
   while 1:
      O.run(1000,True)
      if triaxDone:
         n = porosity()
         # reduce inter-particle friction if void ratio is big
         if n/(1.-n) > e:
            ctrMu *= 0.99
            print ctrMu, n/(1.-n)
            for inter in O.interactions:
               inter.phys.tangensOfFrictionAngle = tan(radians(ctrMu))
            O.materials[spMat].frictionAngle=radians(ctrMu)
            triaxDone = False
         else:
            # now start isotropic compression
            triax.goal = (-conf,-conf,-conf)
            triax.doneHook = "compactionFinished()"
            # set inter-particle friction to correct level
            for inter in O.interactions:
               inter.phys.tangensOfFrictionAngle = tan(radians(mu))
            O.materials[spMat].frictionAngle=radians(mu)
            break
else:
   triax.goal=(-conf,-conf,-conf)
   triax.maxStrainRate=(10.*rate,10.*rate,10.*rate)
   triax.doneHook='compactionFinished()'

def compactionFinished():
   if unbalancedForce()<stabilityRatio:
      # set the current cell configuration to be the reference one
      O.cell.trsf=Matrix3.Identity
      # set loading type: constant pressure in x and y
      triax.goal=(-conf,-conf,-dstrain)
      triax.stressMask=3; triax.globUpdate = 1;
      # allow faster deformation along x,y to better maintain stresses
      triax.maxStrainRate=(10*rate,10*rate,rate)
      # next time, call triaxFinished instead of compactionFinished
      triax.doneHook="addPlotData()"
      # set damping to normal level
      newton.damping = 0.2
      print 'start trixial shearing.'

def addPlotData():
   s = triax.stress
   s33_over_s11 = 2.*s[2]/(s[0]+s[1])
   e_x, e_y, e_z = -triax.strain
   e_v = e_x+e_y+e_z
   n = porosity()
   e = n/(1.-n)
   plot.addData( e_r=100.*(e_x+e_y)/2., s33_over_s11=s33_over_s11, e_v=100.*e_v )
   if abs(e_z-strainGoal)/strainGoal > stabilityRatio:
      triax.goal[2] -= dstrain
   else:
      numpy.save('mcSimulations/'+str(table.key)+'.npy',plot.data)
      print 'triaxial shearing finished.'
      O.pause()

# run in batch mode
O.run()
waitIfBatch()
