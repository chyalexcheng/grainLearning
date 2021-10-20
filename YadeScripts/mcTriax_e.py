# encoding: utf-8

# read parameter values from table
readParamsFromTable(
   E = 1.37459e+11,
   v = 0.2,
   # need to use the same mu_i as in the prepared isoState
   mu_i = 0.14690961144277795,
   mu = 0.36298,
   initPoro = 0.95,
   key = 771,
   kr = 2458.66000,
   mu_r = 0.39885,
)
              
#~ import xlrd
from yade.params import table
from yade import plot
import numpy as np

# set current state as initial
def setInitial():
    O.cell.trsf = Matrix3.Identity
    O.cell.velGrad = Matrix3.Zero
    setRefSe3()
    triax.maxStrainRate = Vector3.Ones*rate
    triax.stressMask = 0
    if debug: 
        print('setInitial')
        #~ O.saveTmp()
    #~ addPlotData()
    startLoading()
        
# start to load the packing
def startLoading():
    if debug: print('startLoading')
    triax.goal = loadData.pop()
    triax.maxUnbalanced = initStabilityRatio
    triax.relStressTol = stressTolRatio
    triax.doneHook = 'calmSystem()'
    newton.damping = lowDamp

# add data into a dictionary    
def addPlotData():
    if debug: print('addPlotData')
    sig_a = triax.stress[2]
    sig_r = 0.5*(triax.stress[0]+triax.stress[1])
    C = avgNumInteractions(skipFree=False)
    CN = avgNumInteractions(skipFree=True)
    overlap = np.mean([inter.geom.penetrationDepth for inter in O.interactions])
    p = (sig_a+2*sig_r)/-3e6
    q = (sig_a-sig_r)/-1e6
    K0 = sig_r/sig_a
    n = porosity()
    e_x, e_y, e_z = -triax.strain
    e_v = e_x+e_y+e_z
    plot.addData(p=p, q=q, n=n, e_x=e_x, e_y=e_y, e_z=e_z, C=C, CN=CN, overlap=overlap, K0=K0)
    if checkWaveSpeed:
        if abs(e_z-checkStrain[-1])/checkStrain[-1]<0.01:
            O.save(stateDir+testName+'%i_%.5e_%.5f_%.5f_%.5f_%.5f'%(table.key,table.E,mu_i,table.mu,table.kr,table.mu_r)+'_%.5f'%(checkStrain.pop())+'.yade.gz')
        if len(checkStrain) == 0:
            O.pause(); exit()

    if len(loadData) != 0: startLoading()
    else: 
        dataName = stateDir+testName+'%i_%.5e_%.5f_%.5f_%.5f_%.5f'%(table.key,table.E,mu_i,table.mu,table.kr,table.mu_r)+'.txt'
        plot.saveDataTxt(dataName); O.pause()
        #~ print('Simulation took %.3f hours'%((O.realtime-t0)/3600))

# switch on high background damping to stablize the packing
def calmSystem():
    if debug: print('calmSystem')
    newton.damping = highDamp
    triax.maxUnbalanced = stabilityRatio
    triax.doneHook = 'addPlotData()'
    
# ouput information for debugging
def debugOut():
    print('unb=%.3e; n=%.3e; p=%.3f; pTarget=%.3f; strainRate=%.3e; Ke=%.3e;'\
    %(unbalancedForce(), porosity(), -triax.stress.sum()/3e6, -triax.goal.sum()/3e6, triax.strainRate.norm(), kineticEnergy()))

##~~~~~~~~~~~~~~##
##    Main script    ##
##~~~~~~~~~~~~~~##

# read external load history
testName = 'VAE3_'
checkWaveSpeed = False
useInitFriction = False
useInitYoung = False
useInitRolling = False
debug = False

# isotropic compression test
if testName[2:] == 'I2_':
    loadFile = 'e'+testName[:-1]+'.txt'
    fin = open(loadFile,'r')
    lines = fin.readlines(); loadData = []
    for l in lines: loadData.append(-Vector3.Ones*float(l)/3.)
    loadData.reverse()

# anisotropic compression test
if testName == 'K02_':
    loadFile = 'e'+testName+'.txt'
    fin = open(loadFile,'r')
    lines = fin.readlines(); loadData = []
    for l in lines: 
        e_r, e_a = l.split()
        loadData.append(-Vector3(float(e_r),float(e_r),float(e_a)))
    loadData.reverse()

# eodometric compressoin test
if testName == 'VAE3_':
    loadFile = 'e'+testName+'full.txt'
    fin = open(loadFile,'r')
    lines = fin.readlines(); loadData = []
    for l in lines: 
        loadData.append(-Vector3(0,0,float(l)))
    loadData.reverse()

# volumetric strain at which wave velocities are checked
if checkWaveSpeed:
    loadFile = testName+'waveCheckStrain.txt'
    fin = open(loadFile,'r')
    lines = fin.readlines(); checkStrain = []
    for l in lines: checkStrain.append(float(l))
    checkStrain.reverse()

# adjust interparticle friction angle to get correct porosity
def checkVolumeFraction(mu_i):
    if porosity() > finalPoro:
        mu_i *= 0.999
        #~ print(mu_i, porosity())
        setContactFriction(mu_i)
        triax.doneHook = ' '
        newton.damping = lowDamp
        O.engines[-2].dead = False
        O.engines[-2].iterLast = O.iter
    else:
        triax.doneHook = 'setInitial()'
        # set real friction angle
        if not useInitFriction: setContactFriction(mu)
        O.engines[-2].dead = True        
    return mu_i
        
# Run on 1: local destop or 2: remote server
run = 1
if run == 1: stateDir = '/storage/cheng/calibration/VA/ctExtra/isoStates/VE2/'
if run == 2: stateDir = './VA/ctExtra/isoStates/VE2/'

# check whether a state is saved

fileName = stateDir+'isoState_1_7.07600e+10_0.28600_0.95000_1.16505e+04_0.13375.yade.gz'
#~ fileName = stateDir+'isoState'+'_%i'%(table.key)+'_%.5e'%(table.E)+'_%.5f'%(table.mu_i)+'.yade.gz'
#~ if table.initPoro != 0:    fileName = fileName[:-8]+'_%.5f'%(table.initPoro)+'.yade.gz'
#~ if table.kr != 0 or table.mu_r != 0: fileName = fileName[:-8]+'_%.5e_%.5f'%(table.kr,table.mu_r)+'.yade.gz'

if not os.path.exists(fileName): 
    print('state does not exist'); exit()

# glass bead parameters (units: ug->1e-9kg; mm->1e-3m; ms->1e-3s)
lenScale = 1e3                   # lenth in mm <- 1e-3 m
sigScale = 1                     # stress in ug/(mm*ms^2) <- Pa
rhoScale = 1                     # density in ug/mm^3 <- kg/m^3
keScale = 1e9                    # energy in ug*mm^2/ms^2 <- nJ
E = table.E*sigScale
v = table.v
mu_i = table.mu_i
mu = table.mu
rho = 2465.9*rhoScale
n0 = 0.39613 
    
O.load(fileName)

# simulation control
rate = 0.1                         # strain rate
lowDamp = 0.2; highDamp = 0.9    # damping coefficient
initStabilityRatio = 1.e-3       # initial stability threshold
initStressTolRatio = 5.e-3       # initial tolerance ratio
stabilityRatio = 1.e-5           # stability threshold
stressTolRatio = 1.e-3           # stress tolerance ratio
finalPoro = 0.3910585819

#~ # stablize packing after set real friction angle and Poisson's ratio
#~ 
#~ if testName != 'VBI2_': dataFileName = testName[:-1]+' output_noreset.xls'
#~ else: dataFileName = testName[:-1]+' output.xls'
#~ wb = xlrd.open_workbook(dataFileName)
#~ sh = wb.sheet_by_index(0)
#~ q = sh.col_values(1)[3]
#~ p = sh.col_values(2)[3]
#~ 
#~ if testName == 'VAI2_' and table.initPoro == 0:    
    #~ O.engines = O.engines[:-1]+[
                #~ PeriTriaxController(label='triax',
                #~ # whether they are strains or stresses
                #~ stressMask = 7,
                #~ # type of servo-control
                #~ dynCell = True,
                #~ # wait until the unbalanced force goes below this value
                #~ maxStrainRate= Vector3.Ones*rate)]+O.engines[-1:]
    #~ O.engines[-1].dead = True
    #~ triax = O.engines[-2]
    #~ q = 0.1113213874; p = 5.0441837355
#~ 
#~ sig_a = -(p + 2.*q/3)*1e6
#~ sig_r = -(p -    q/3)*1e6
#~ triax.goal = Vector3(sig_r,sig_r,sig_a)
#~ triax.stressMask = 7
#~ triax.maxUnbalanced = stabilityRatio
#~ triax.relStressTol = initStressTolRatio
#~ triax.absStressTol = abs(triax.relStressTol*triax.goal.sum()/3)
#~ newton.damping = highDamp
#~ O.cell.trsf = Matrix3.Identity; O.cell.velGrad = Matrix3.Zero; setRefSe3()
#~ triax.doneHook = 'setInitial()'
#~ if debug: O.engines = O.engines[:]+[PyRunner(command='debugOut()',iterPeriod=5000)]
#~ # set contact damping to zero
#~ if table.initPoro != 0:
    #~ O.engines[2].physDispatcher.functors[0].betas.val = 0.0
    #~ O.engines[2].physDispatcher.functors[0].betan.val = 0.0

# change Young's modulus
if not useInitYoung:
    O.materials[0].young = E
    O.interactions.clear()
# change rolling parameters
if not useInitRolling:
    O.engines[2].physDispatcher.functors[0].krot = abs(table.kr)
    O.engines[2].physDispatcher.functors[0].ktwist = abs(table.kr)
    O.engines[2].physDispatcher.functors[0].eta = abs(table.mu_r)
    O.engines[2].lawDispatcher.functors[0].includeMoment = True
    O.interactions.clear()
# change initial friction
if not useInitFriction: setContactFriction(O.materials[0].frictionAngle)


O.engines[-1].command = 'debugOut()'
if debug: O.engines[-1].dead = False
    
# reduce mu_i if porosity is big
O.engines[-2].command = "triax.doneHook='mu_i=checkVolumeFraction(mu_i); newton.damping=highDamp'; O.engines[-2].dead=True"
O.engines[-2].dead = True
newton.damping = 0.2

# run in batch mode
#~ setInitial()
O.run()
waitIfBatch()

#~ # tune safety factor of time stepper
#~ O.engines[3].timestepSafetyCoefficient = 0.2
