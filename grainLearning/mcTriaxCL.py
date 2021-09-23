# encoding: utf-8
readParamsFromTable(
   E = 8.736,
   v = 0.08,
   kr = 0.564,
   eta = 0.194,
   mu = 30.203,
   num = 10000,
   conf = 0.2e6,
   key = 1,
   mode = 'drained',
   unknownOk=True
)

import glob, os
import numpy as np
from yade.params import table
from yade import pack, plot
from pathlib import Path

# function to sum up geometrical fabric information
def getGeoFabric(inter):
	n = inter.geom.normal
	fab_i = Matrix3.Zero
	for i in range(3):
		for j in range(3):
			fab_i[i,j] = n[i]*n[j]
	return fab_i

# function to sum up geometrical fabric information
def getMechFabric(inter,a_c):
	fab_n_i = Matrix3.Zero
	fab_t_i = Matrix3.Zero
	n = inter.geom.normal
	t = inter.phys.shearForce/inter.phys.shearForce.norm()
	# assign tangential unit vector to zero if no slip
	if isnan(t[0]): t = Vector3.Zero		
	fn = inter.phys.normalForce.norm()
	ft = inter.phys.shearForce.norm()
	# double dot product of a_c and fab_i
	Ddot = sum([a_c[i,j]*n[i]*n[j] for i in range(3) for j in range(3)])
	for i in range(3):
		for j in range(3):
			fab_n_i[i,j] = fn*n[i]*n[j]
			fab_t_i[i,j] = ft*t[i]*n[j]
	fab_n_i /= (1.+Ddot)
	fab_t_i /= (1.+Ddot)
	return fab_n_i, fab_t_i

# functions for micromechanical analysis
def getFabricVariables(s_dev):
	# compute fabric tensor fab and fabric anisotropy tensor a_c
	fab = Matrix3.Zero
	count = 0
	## Sum up information from interactions
	for inter in O.interactions:
		# overall fabric (exclude inters of wirePhys)
		if inter.isReal:
			fab += getGeoFabric(inter)
			count += 1		

	# Average over number of interactions
	fab /= count
	# fabric anisotropy tensor
	a_c = 15./2.*(fab - Matrix3.Identity*fab.trace()/3.)
	
	## Compute mechanical fabric tensor a_n and a_t
	# normal force
	fab_n = Matrix3.Zero
	# tangential force
	fab_t = Matrix3.Zero
	
	# Sum up information from interactions
	for inter in O.interactions:
		# overall fabric (exclude inters of wirePhys)
		if inter.isReal:
			n_i, t_i = getMechFabric(inter,a_c)
			fab_n += n_i
			fab_t += t_i

	## Average over number of interactions
	# normal force
	fab_n /= count
	# tangential force
	fab_t /= count
	
	## Mechanical anisotropy fabric tensors of each type
	# average contact normal force
	f0 = fab_n.trace()
	# normal force
	a_n = 15./2.*(fab_n - Matrix3.Identity*f0/3)/f0 
	# tangential force
	a_t = 15./3.*(fab_t - Matrix3.Identity*fab_t.trace()/3.)/f0 
	
	## Compute anisotropy indicators see https://www.icevirtuallibrary.com/doi/abs/10.1680/geot.12.P.040
	# overall (exclude inters of wirePhys)
	DdotC = sum([a_c[i,j]**2. for i in range(3) for j in range(3)])
	DdotN = sum([a_n[i,j]**2. for i in range(3) for j in range(3)])
	DdotT = sum([a_t[i,j]**2. for i in range(3) for j in range(3)])	
	DdotS_DEV = sum([s_dev[i,j]**2. for i in range(3) for j in range(3)])
	SrC = sum([a_c[i,j]*s_dev[i][j] for i in range(3) for j in range(3)])/sqrt(DdotC)/sqrt(DdotS_DEV)
	SrN = sum([a_n[i,j]*s_dev[i][j] for i in range(3) for j in range(3)])/sqrt(DdotN)/sqrt(DdotS_DEV)
	SrT = sum([a_t[i,j]*s_dev[i][j] for i in range(3) for j in range(3)])/sqrt(DdotT)/sqrt(DdotS_DEV)
	AnIso_C = SrC/abs(SrC)*sqrt(3./2.*DdotC)
	AnIso_N = SrN/abs(SrN)*sqrt(3./2.*DdotN)
	AnIso_T = SrT/abs(SrT)*sqrt(3./2.*DdotT)
	
	## compute joint invariant K of dev(a_c) and s_dev
	# overall
	K = (a_c*s_dev).trace()
	n_a_c = a_c/sqrt(DdotC)
	n_s_dev = s_dev/sqrt(DdotS_DEV)
	A = sum([n_a_c[i,j]*n_s_dev[i,j] for i in range(3) for j in range(3)])

	return f0, AnIso_C, AnIso_N, AnIso_T, K, A

# Simulation control
debug = False
random = False            		# use ramdom particle packing or not
num = table.num          		# number of soil particles
dScaling = 1e0         		    # density scaling  (treat it as a parameter too?)
e = 0.68              		    # initial void ratio
conf = table.conf     		    # confining pressure
rate = 0.1            		    # loading rate (strain rate) (decrease this for serious calculations)
damp = 0.2                		# damping coefficient
initStabilityRatio = 1.e-3		# initial stability threshold
stabilityRatio = 1.e-6    		# stability threshold
stressTolRatio = 1.e-3    		# stress tolerance ratio
lowDamp = 0.2					# low and high damping values
highDamp = 0.9    		  
strainGoal = 0.20 		        # target strain level
nLoadSteps = 200				# number of loading steps
dstrain = strainGoal/nLoadSteps	# strain increment

# load strain/stress data for quasi-static loading
if table.mode == 'drained':
	loadData = [Vector3(-conf,-conf,-dstrain*i) for i in range(nLoadSteps)]
elif table.mode == 'undrained':
	loadData = [Vector3(0.5*dstrain*i,0.5*dstrain*i,-dstrain*i) for i in range(nLoadSteps)]
loadData.reverse()

# corners to define specimen size
mn,mx=Vector3.Zero,Vector3(0.1,0.1,0.2)

# Soil sphere parameters
E=10**table.E		      # micro Young's modulus
v=table.v                 # micro Poisson's ratio
kr=table.kr               # rolling/bending stiffness
eta=table.eta             # rolling/bending plastic limit
mu = table.mu             # contact friction during shear
ctrMu = table.mu          # use small mu to prepare dense packing?
rho = 2650*dScaling       # soil density

# create materials
spMat = O.materials.append(
   CohFrictMat(young=E,poisson=v,frictionAngle=radians(mu),density=rho,isCohesive=False,
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
   print("cellSize= ",O.cell.hSize)
   sp.save('RandomPacks/PeriSp_'+str(num)+'_0.68.txt')
else:
   if num==1000: O.cell.hSize=Matrix3(0.04622,0,0, 0,0.04612,0, 0,0,0.09212)
   if num==2000: O.cell.hSize=Matrix3(0.05013,0,0, 0,0.05015,0, 0,0,0.09945)
   if num==5000: O.cell.hSize=Matrix3(0.04617,0,0, 0,0.04626,0, 0,0,0.09201)
   if num==8000:O.cell.hSize=Matrix3(0.0462201,0,0, 0,0.0461205,0, 0,0,0.0921203)
   if num==10000:O.cell.hSize=Matrix3(0.04609,0,0, 0,0.04616,0, 0,0,0.09221)
   if num==27000:O.cell.hSize=Matrix3(0.0462201,0,0, 0,0.0461205,0, 0,0,0.0921203)
   print('Loading configuration from Packs/PeriSp_'+str(num)+'_0.68.txt')
   sp.load('Packs/PeriSp_'+str(num)+'_0.68.txt')

# load spheres to simulation
spIds=sp.toSimulation(material=spMat)

# yade data directory
yadeDataDir = 'triax/CL/%.1fe6/'%(conf/1e6) + table.mode +'/'
path = Path(yadeDataDir)
path.mkdir(parents=True, exist_ok=True)
print('yade data directory already exists (%i files)\n' % len(glob.glob(yadeDataDir + '/*')))
if os.path.exists(yadeDataDir+'/SimData_'+table.mode+'_%i'%(table.key)+'.npy'): exit()

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
   PyRunner(command="checkUnjammed()",iterPeriod=10000,dead=False),
   ]

# prepare dense particle packing (TODO)
if random:
   triaxDone = False
   triax.goal=(-1e3,-1e3,-1e3)
   triax.maxStrainRate=(10.*rate,10.*rate,10.*rate)
   triax.doneHook="triaxDone=True;newton.damping=highDamp"
   # prepare dense packing
   while True:
      O.run(1000,True)
      if triaxDone:
         n = porosity()
         # reduce inter-particle friction if e is still big
         if n/(1.-n) > e:
            ctrMu *= 0.99
            print(ctrMu, n/(1.-n))
            setContactFriction(ctrMu)
            triaxDone = False
         else:
            # now start isotropic compression
            triax.goal = (-conf,-conf,-conf)
            triax.doneHook = "compactionFinished()"
            # set inter-particle friction to correct level
            setContactFriction(mu)
            break
else:
   triax.goal=(-conf,-conf,-conf)
   triax.maxStrainRate=(10.*rate,10.*rate,10.*rate)
   triax.doneHook='compactionFinished()'

# set the reference configuration to the current and save the initial state
def compactionFinished():
   if unbalancedForce()<stabilityRatio:
      if debug: print('compaction finished')
      # set the current cell configuration to be the reference one
      O.cell.trsf = Matrix3.Identity
      O.cell.velGrad = Matrix3.Zero
      setRefSe3()
      # set loading moade: drained or undrained
      triax.globUpdate = 1
      if table.mode == 'drained':
          triax.stressMask=3
          # allow faster deformation along x,y to better maintain stresses
          triax.maxStrainRate=(10*rate,10*rate,rate)
      elif table.mode == 'undrained':
          triax.stressMask=0
          # allow faster deformation along x,y to better maintain stresses
          triax.maxStrainRate=(0.5*rate,0.5*rate,rate)
      # next time, call addPlotData instead of compactionFinished
      triax.doneHook="addPlotData()"
      # set damping to a normal value
      newton.damping = lowDamp
      print('start trixial shearing.')
      # save the initial state and start loading
      O.save(yadeDataDir+'/simState_'+table.mode+'_%i_%i'%(table.key,0)+'.yade.gz')
      startLoading()

# periodically check if the packing is unjammed under constant volume
def checkUnjammed():
	s = -getStress()
	p = s.trace()/3.0
	s_dev = s - Matrix3.Identity*p
	DdotS_DEV = sum([s_dev[i,j]**2. for i in range(3) for j in range(3)])
	q = sqrt(3./2.*DdotS_DEV)
	if p < 10 and q < 10:
		print('The system is unjammed! Simulation will be stopped now.')
		# save simulation data and parameters
		addPlotData()
		params = {}
		for name in table.__all__: params[name] = eval('table.'+name)
		np.save(yadeDataDir+'/SimData_'+table.mode+'_%i'%(table.key)+'.npy',tuple([params,plot.data]))
		print('triaxial shearing finished after %.3f hours'%((O.realtime-t0)/3600))
		O.pause(); exit()

# start to load the packing
def startLoading():
	if debug: print('startLoading')
	triax.goal = loadData.pop()
	triax.maxUnbalanced = initStabilityRatio
	triax.doneHook = 'calmSystem()'
	newton.damping = lowDamp

# switch on high background damping to stablize the packing
def calmSystem():
	if debug: print('calmSystem')
	newton.damping = highDamp
	triax.maxUnbalanced = stabilityRatio
	# ~ triax.doneHook = 'saveSimState(); addPlotData()'
	triax.doneHook = 'addPlotData()'

def saveSimState():
	n = nLoadSteps-len(loadData)
	if not n%10:
		O.save(yadeDataDir+'/simState_'+table.mode+'_%i_%i'%(table.key,n)+'.yade.gz')
	
def addPlotData():
	if debug: print('addPlotData')  

	s = -getStress()
	p = s.trace()/3.0
	s_dev = s - Matrix3.Identity*p
	DdotS_DEV = sum([s_dev[i,j]**2. for i in range(3) for j in range(3)])
	q = sqrt(3./2.*DdotS_DEV)
	e_x, e_y, e_z = -triax.strain
	e_v = e_x+e_y+e_z
	n = porosity()
	e = n/(1.-n)
	
	# return anisotropy
	# n : normal interaction
	# t : tangential interaction
	f0, a_c, a_n, a_t, K, A = getFabricVariables(s_dev)
	
	print("Triax goal is: ",triax.goal, "dt=" ,O.dt ,"numIter=",O.iter,"real time=", O.realtime ) 
	plot.addData(e=e, e_v=e_v, e_x=e_x, e_y=e_y, e_z=e_z, p=p, q=q,\
		f0=f0, a_c=a_c, a_n=a_n, a_t=a_t, K=K, A=A, dt=O.dt, numIter=O.iter)
	
	# continue loading if loadData is not empty
	if len(loadData) !=0: startLoading()
	else:
		# save simulation data and parameters
		params = {}
		for name in table.__all__: params[name] = eval('table.'+name)
		np.save(yadeDataDir+'/SimData_'+table.mode+'_%i'%(table.key)+'.npy',tuple([params,plot.data]))
		print('triaxial shearing finished after %.3f hours'%((O.realtime-t0)/3600))
		O.pause(); exit()

# run in batch mode
t0 = O.realtime
O.run()
waitIfBatch()
