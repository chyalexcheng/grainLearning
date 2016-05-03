# encoding: utf-8

readParamsFromTable(
   E = 4e9,
   v = 0.33,
   kr = 0.13,
   eta = 0.13,
   mu = 29.0,
   num = 1000,
   conf = 0.2e6,
   key = 1,
   unknownOk=True
)

from yade.params import table
from yade import pack, plot

# Simulation control
random = False            # use ramdom particle packing or not
num = table.num           # number of soil particles
dScaling = 1e3            # density scaling
e = 0.68                  # initial void ratio
conf = table.conf         # confining pressure
strainGoal = 0.15         # target strain level
dstrain = strainGoal/150  # strain increment
rate = 0.1                # loading rate (strain rate)
damp = 0.2                # damping coefficient
stressTolRatio = 1.e-4    # tolerance for stress goal
stabilityRatio = 1.e-3    # threshold for quasi-static condition
# corners to define specimen size
mn,mx=Vector3.Zero,Vector3(0.1,0.1,0.2)

# Soil sphere parameters
E=table.E                 # micro Young's modulus
v=table.v                 # micro Poisson's ratio
kr=table.kr               # rolling/bending stiffness
eta=table.eta             # rolling/bending plastic limit
mu = table.mu             # contact friction during shear
ctrMu = table.mu          # use small mu to prepare dense packing?
rho = 2650*dScaling       # soil density

# create materials
spMat = O.materials.append(
   CohFrictMat(young=E,poisson=v,frictionAngle=radians(ctrMu),density=rho,isCohesive=False,
      alphaKr=kr,alphaKtw=kr,momentRotationLaw=True,etaRoll=eta,etaTwist=eta))
	
# create a cloud of ramdomly positioned spheres
O.periodic = True
sp=pack.SpherePack()
if random:
   sizes=[.00575,.00685,.00816,.00969,.01150,.01369,.01626]
   cumm=[.013,.021,.058,.174,.811,.927,1]
   sp.makeCloud(minCorner=mn,maxCorner=mx,psdSizes=sizes,psdCumm=cumm,\
      distributeMass=True,porosity=e/(1+e),seed=1,num=num)
   O.cell.hSize = Matrix3(mx[0],0,0, 0,mx[1],0, 0,0,mx[2])
else:
   if num==1000: O.cell.hSize=Matrix3(0.04622011418432206,0,0, 0,0.04612047733502004,0, 0,0,0.09212029433303306)
   if num==2000: O.cell.hSize=Matrix3(0.050134992262392535,0,0, 0,0.050157098506083464,0, 0,0,0.0994518551339961)
   if num==5000: O.cell.hSize=Matrix3(0.046170983201952086,0,0, 0,0.04625783531203574,0, 0,0,0.09201499300536521)
   if num==10000:O.cell.hSize=Matrix3(0.046093488302021875,0,0, 0,0.04616421348377587,0, 0,0,0.09220693508372971)
   sp.load('PeriSp_'+str(num)+'_0.68.txt')

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

# prepare dense particle packing
if random:
   triaxDone = False
   triax.goal=(-0.1*conf,-0.1*conf,-0.1*conf)
   triax.maxStrainRate=(10.*rate,10.*rate,10.*rate)
   triax.doneHook="triaxDone=True;newton.damping=0.9"
   # prepare dense packing
   while 1:
      O.run(100,True)
      if triaxDone:
         n = porosity()
         # reduce inter-particle friction if e is still big
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
      # set loading type: constant pressure in x,y, 8.5% compression in z
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
   plot.addData( e_r=100.*(e_x+e_y)/2., e_a=100.*e_z, e_v=100.*e_v, e=e, s33_over_s11=s33_over_s11)
   if abs(e_z-strainGoal)/strainGoal > stabilityRatio:
      triax.goal[2] -= dstrain
   else:
      numpy.save('./mcSimulations/CL/'+str(table.num)+'/'+'%3.1f'%(table.conf/1e6)+'/'+str(table.key)+'.npy',plot.data)
      numpy.save('./covResults/CL'+str(table.num)+'/'+'%3.1f'%(table.conf/1e6)+'_'+str(table.key)+'.npy',plot.data)
      print 'triaxial shearing finished.'
      O.pause()

# run in batch mode
O.run()
waitIfBatch()
