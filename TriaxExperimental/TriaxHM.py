# encoding: utf-8
readParamsFromTable(
#   E_m = 9.61521,
#   v = 0.064,
#   kr = 0.148,
#   eta = 0.202,
#   mu = 30.387,
#   num = 1000,
#   conf = 0.2e6,
   
#   E_m = 8.7695,
#   v = 0.304,
#   kr = 0.344,
#   eta = 0.567,
#   mu = 24.027,
#   num = 1000,
#   conf = 0.5e6,
   E_m = 1.000000000000000000e+01,
   v = 5.555555555600000250e-02,
   kr = 5.999999999999999778e-01,
   eta = 4.285714285700000237e-01,
   mu = 1.636363636400000132e+01,
   num = 10000,
   conf = 0.5e6,
#   E_m = 7.5,
#   v = 0.222222222222222,
#   kr = 0.8,
#   eta = 0.57142857143,
#   mu = 0.21818181818,
#   num = 10000,
#   conf = 0.5e6,
   
#   E_m = 9.9203,
#   v = 0.049,
#   kr = 0.363,
#   eta = 0.48,
#   mu = 24.951,
#   num = 1000,
#   conf = 1.0e6,
   
#   E_m = 9.83206,
#   v = 0.482,
#   kr = 0.998,
#   eta = 0.171,
#   mu = 20.06,
#   num = 1000,
#   conf = 2.0e6,
   
   unknownOk=True
)
import glob, os
import numpy as np
from yade.params import table
from yade import pack, plot

isBatch = runningInBatch()	# check if run in batch mode
if isBatch:
	print('Running: '+O.tags['description'])
    
tsc = 0.8

# Simulation control
debug = False
random = False            # use ramdom particle packing or not
num = table.num           # number of soil particles
dScaling = 1e3            # density scaling  (treat it as a parameter too?)
e = 0.68                  # initial void ratio
conf = table.conf         # confining pressure
rate = 0.1                # loading rate (strain rate) (decrease this for serious calculations)
damp = 0.2                # damping coefficient
stabilityRatio = 1.e-3    # threshold for quasi-static condition (decrease this for serious calculations)
stressTolRatio = 1.e-3    # tolerance for stress goal
initStabilityRatio = 1.e-3# initial stability threshold
obsCtrl = 'e_z'			  # key for simulation control
lowDamp = 0.2			  # damping coefficient
highDamp = 0.9    		  

# load strain/stress data for quasi-static loading
baseFolder ="../"
obsFile ="Test_"+"%.1f.data"%(conf/1e6)
data = np.loadtxt(obsFile,skiprows=1)
fOpen = open(obsFile,'r')
firstLine = fOpen.readlines(1)[0]
simDataKeys = firstLine[:-1].split('\t\t')[1:]
#print(simDataKeys)
obsCtrlID = simDataKeys.index(obsCtrl)
#print(obsCtrlID)
#print(data)
#exit()
loadData = [Vector3(-conf,-conf,-data[i,obsCtrlID]*0.01) for i in range(data.shape[0])]
loadData.reverse()
print('Use '+simDataKeys.pop(obsCtrlID)+\
	' to control quasi-static loading and output ' +' '.join(simDataKeys))

# corners to define specimen size
mn,mx=Vector3.Zero,Vector3(0.1,0.1,0.2)

# Soil sphere parameters
E=pow(10,table.E_m)          # micro Young's modulus
print("Youngs modulus is ", E)
print("Exponent is ", table.E_m)
v=table.v                 # micro Poisson's ratio
kr=table.kr               # rolling/bending stiffness
eta=table.eta             # rolling/bending plastic limit
mu = table.mu             # contact friction during shear
ctrMu = table.mu          # use small mu to prepare dense packing?
rho = 2650*dScaling       # soil density

# create materials
spMat = O.materials.append(
   FrictMat(young=E,poisson=v,frictionAngle=radians(ctrMu),density=rho))
	
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
   # safe sphere package 
   if 1==1:
    sp.save(baseFolder+'Packs/PeriSp_'+str(num)+'_0.68.txt')
else:
   if num==1000: O.cell.hSize=Matrix3(0.04622,0,0, 0,0.04612,0, 0,0,0.09212)
   if num==2000: O.cell.hSize=Matrix3(0.05013,0,0, 0,0.05015,0, 0,0,0.09945)
   if num==5000: O.cell.hSize=Matrix3(0.04617,0,0, 0,0.04626,0, 0,0,0.09201)
   if num==8000:O.cell.hSize=Matrix3(0.0462201,0,0, 0,0.0461205,0, 0,0,0.0921203)
   if num==10000:O.cell.hSize=Matrix3(0.04609,0,0, 0,0.04616,0, 0,0,0.09221)
   if num==27000:O.cell.hSize=Matrix3(0.0462201,0,0, 0,0.0461205,0, 0,0,0.0921203)
   # add own packages
   if num==20000:O.cell.hSize=Matrix3(0.1,0,0, 0,0.1,0, 0,0,0.2)
   print('PeriSp_'+str(num)+'_0.68.txt')
   sp.load(baseFolder+'Packs/PeriSp_'+str(num)+'_0.68.txt')

# load spheres to simulation
spIds=sp.toSimulation(material=spMat)

# Definition of time step safety coefficient  
def getTimeStepSafetyCoefficient():
    m = table.E_m
    if(m>=10):
        return 0.8*0.8
    if(7<=m<8):
        if(kr>0.1):
            return 0.025
        else:
            return 0.1
    if(8<=m<9):
        if(kr>0.2):
            return 0.1
        else:
            return 0.2
    if(9<=m<10):
        if(kr<0.2):
            return 0.8*0.8
        elif(0.2<=kr<0.6):
            return 0.4*0.8
        else:
            return 0.3*0.8  

    
tsc = getTimeStepSafetyCoefficient()
print("Time step safety coefficient is: ",tsc)

# define engines
O.engines=[
   ForceResetter(),
   InsertionSortCollider([Bo1_Sphere_Aabb()]),
   InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_MindlinPhys(
            krot=  table.kr,
            ktwist= table.kr,
            eta=table.eta
        )],
   [Law2_ScGeom_MindlinPhys_Mindlin(includeMoment=True)]
   ),
   GlobalStiffnessTimeStepper(timestepSafetyCoefficient=tsc),
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
   triax.doneHook="triaxDone=True;newton.damping=highDamp"
   # prepare dense packing
   while 1:
      O.run(100,True)
      if triaxDone:
         n = porosity()
         # reduce inter-particle friction if e is still big
         if n/(1.-n) > e:
            ctrMu *= 0.99
            print(ctrMu, n/(1.-n))
            for inter in O.interactions:
               inter.phys.tangensOfFrictionAngle = tan(radians(ctrMu))
            O.materials[spMat].frictionAngle=radians(ctrMu)
            triaxDone = False
         else:
            # now start isotropic compression
            triax.goal = (-conf,-conf,-conf)
            triax.doneHook = "compaction finished()"
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
      if debug: print('compaction finished')
      # set the current cell configuration to be the reference one
      O.cell.trsf = Matrix3.Identity
      O.cell.velGrad = Matrix3.Zero
      setRefSe3()
      # set loading type: constant pressure in x,y, 8.5% compression in z
      triax.stressMask=3; triax.globUpdate=1;
      # allow faster deformation along x,y to better maintain stresses
      triax.maxStrainRate=(10*rate,10*rate,rate)
      # set damping to a normal value
      newton.damping = lowDamp
      
      #write .txt file
      ratio = int(O.iter/O.realtime)
      
      if isBatch:
           dataName = O.tags['description'].split('-')[1]+'.txt'
      else:
           dataName = 'SampleTest00.txt'
      np.savetxt(dataName, [ratio])
      print('start trixial shearing.')
      startLoading()

# start to load the packing
def startLoading():
	global loadData
	if debug: print('startLoading')
	triax.goal = loadData.pop()
	triax.maxUnbalanced = initStabilityRatio
	triax.relStressTol = stressTolRatio
	triax.doneHook = 'calmSystem()'
	newton.damping = lowDamp

# switch on high background damping to stablize the packing
def calmSystem():
	if debug: print('calmSystem')
	newton.damping = highDamp
	triax.maxUnbalanced = stabilityRatio
	triax.doneHook = 'addPlotData()'

def addPlotData():
	global loadData
	if debug: print('addPlotData')  
	s = triax.stress
	s33_over_s11 = 2.*s[2]/(s[0]+s[1])
	e_x, e_y, e_z = -triax.strain
	e_v = e_x+e_y+e_z
	n = porosity()
	print("Triax goal is: ",triax.goal, "dt=" ,O.dt ,"numIter=",O.iter,"real time=", O.realtime, "Average iterations per second=", O.iter/O.realtime) #, "porosity= ", n )  #e = n/(1.-n)
	#dt=O.dt, numIter=O.iter, realTime = O.realtime 
	plot.addData(e_v =100.*e_v, e_x=100.*e_x, e_y=100.*e_y, e_z=100.*e_z, s33_over_s11=s33_over_s11,dt=O.dt, numIter=O.iter, realTime = O.realtime)
	#plot.addData( e_r=100.*(e_x+e_y)/2., e_a=100.*e_z, e_v=100.*e_v, e=e, s33_over_s11=s33_over_s11)
	if len(loadData) !=0: startLoading()
	else:
            if isBatch:
                dataName = 'triax_'+ O.tags['description'] + '.npy'
            else:
                dataName = 'TriaxHM_'+str(num)+'_Test.npy'
                #dataName = 'triax_%i_%.10e_%.10e' % (table.key, table.E, table.v) + '.npy'
            np.save(dataName,plot.data)
		#dataName = 'triax_%i_%.10e_%.10e' % (table.key, table.E, table.v) + '.txt'
		#plot.saveDataTxt(dataName)
            print('triaxial shearing finished after %.3f hours'%((O.realtime-t0)/3600))
            O.pause()

# run in batch mode
t0 = O.realtime
O.run()
waitIfBatch()
#plot.plots = {'e_z':['s33_over_s11','e_v']}
