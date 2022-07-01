#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld as uw
from underworld import function as fn
#import UWGeodynamics as GEO
#from underworld.scaling import units as u
#from underworld.scaling import dimensionalise, non_dimensionalise
import math
import numpy as np
import underworld.visualisation as glucifer
#import glucifer
import os

rank = uw.mpi.rank


# ## Basic Model Parameters and Scaling

# In[2]:


KL_meters    = 30000. #* u.meter
K_viscosity  = 1.0e14 #* u.pascal * u.second
K_density    = 916.8 #* u.kilogram / (u.meter**3)
Kt_seconds   = 30000. ** 2 / 1.4e-6
KM_kilograms = K_viscosity * KL_meters * Kt_seconds

K_substance  = 1. #* u.mole
KT_degrees   = 173. #* u.kelvin
K_pascal     = K_viscosity / Kt_seconds

print('KM_kilograms  = {}'.format(KM_kilograms))
print('Kt_seconds  = {}'.format(Kt_seconds))
print('K_pascal = {}'.format(K_pascal))


# In[3]:


Yres      = 120
Xres      = int(Yres * 3)
boxHeight = 30000. #* u.meter #33. * u.kilometer
boxLength = boxHeight * 3.
Tmin      = 100. #* u.kelvin
Tmax      = 273. #* u.kelvin
Tc        = 250. #* u.kelvin
deltaT    = Tmax - Tmin
deltaTc   = Tc - Tmin


# In[4]:


K = 1.4e-6 #* (u.meter**2) * (u.second**(-1)) #Diffusivity

rho = 916.8 #* u.kilogram / u.meter**3 #Density

maxRho = 999.0 #maximum density difference

DelRho = maxRho - rho

rhoSalt = 1444. #* u.kilogram / u.meter**3

alpha = 1.56e-4 #/ u.kelvin #Thermal Compressibility

g = 1.315 #* u.meter / u.second**2
# Compressibility in /bar: placeholder value at ~250 K from Marion & Jakubawski (2004)
beta = 2.5e-4 #/ u.bar

Beta = 9.4

etaref = 1.0e14 #* u.pascal * u.second

etamax = 1.0e20 #* u.pascal * u.second

Y = 0.9 #* u.gigapascal #Young's Modulus

G = 3.6 #* u.gigapascal #Shear Modulus

sigy = 1.0e6 #* u.pascal  #Yield Stress

Ra = (rho * alpha * g * deltaT * boxHeight ** 3) / (etaref * K)

Rb = (DelRho * g * boxHeight ** 3) / (etaref * K)


# In[5]:


print('rho  = {}'.format(rho))
print('alpha  = {}'.format(alpha))
print('g  = {}'.format(g))
print('DeltaT  = {}'.format(deltaT))
print('b^3  = {}'.format(boxHeight ** 3))
print('K  = {}'.format(K))
print('eta  = {}'.format(etaref))
print('Ra  = {}'.format(Ra))
print('Rb  = {}'.format(Rb))


# In[6]:


g_s = g / (KL_meters / Kt_seconds ** 2)
rho_s = rho / K_density
DelRho_s = DelRho / rho
rhoSalt_s = rhoSalt / K_density
alpha_s = alpha * KT_degrees
etaref_s = etaref / K_viscosity
etamax_s = etamax / K_viscosity
K_s = K / (KL_meters ** 2 / Kt_seconds)
Lx = boxLength / KL_meters 
Ly = boxHeight / KL_meters #non_dimensionalise(boxHeight)
Tmin_s1 = Tmin / KT_degrees
Tmax_s1 = Tmax / KT_degrees
Tmin_s = (Tmin-Tmin)/(Tmax-Tmin)
Tmax_s = (Tmax-Tmin)/(Tmax-Tmin)
Tc_s = (Tc - Tmin)/(Tmax-Tmin)
#beta_s = non_dimensionalise(beta)
sigy_s = sigy / K_pascal

Ra_s = (rho_s * alpha_s * g_s * (Tmax_s1-Tmin_s1) * Ly ** 3) / (etaref_s * K_s)


# In[7]:


print('rho_s  = {}'.format(rho_s))
print('alpha_s  = {}'.format(alpha_s))
print('g_s  = {}'.format(g_s))
print('K_s  = {}'.format(K_s))
print('eta_s  = {}'.format(etaref_s))
print('Ra_s  = {}'.format(Ra_s))
print('sigy_s = {}'.format(sigy_s))
print('DelRho_s  = {}'.format(DelRho_s))


# ## Flags, Switches, and Variables: Change These!

# In[8]:


#Flags and Switches

LoadFromFile = False
Symmetry     = True #If True, uses symmetry BCs to cut the model domain in half.

Testing      = False #If True, will run model for 10 timesteps. If false, will run for 5000 or 2500 timesteps.
Long         = True  #If true, will run model for 5000 timesteps (as long as Testing = False). Otherwise, 2500 timesteps.

SaltDensity  = False # Change flag to True if chemical density is driven by salt fraction in ice. False prescribes an arbitrary
                     # Density difference.
    
BuffoHiSalt  = False #Activates depth-dependent hi-salt-content profile from Buffo et al. 2020 (Not yet integrated)
BuffoLoSalt  = False #Activates depth-dependent lo-salt-content profile from Buffo et al. 2020 (Not yet integrated)

CircleKeel   = False #Changes the keel from a circular (True) to rectangular (False) shape.

Taper        = False  #Generates a tapered keel, with thickness = BC at far end and thickness = KD at center. 
                      #Note: CircleKeel must be False for taper to generate correctly.
    
LensPresence = False #toggles the presence of a modified ice lens under the keel.


# In[9]:


#Batch Job Info, length of run
Batch = 1  #Batch number
Trial = 1  #for batch submitting jobs. Labels output files with trial number.


# In[10]:


#Domain Variables: Change these numbers to rearrange the size and position of crust, keel, and lens.

CrustThickness = 3000.# meters

KeelRadius = 10000.# meters For circular keels (CircleKeel = True)

KeelWidth = 15000. #meters For rectangular or tapered keels (CircleKeel = False)

KeelWidth2 = 5000. #For symmetry = False. Width of RHS of Keel.

KeelDepth = 5000. #meters For rectangular keels (CircleKeel = False)

LensGap = 0.# * meters #distance between bottom of keel and top of lens

LensWidth = 2. * KeelRadius #Half-width of lens

LensHeight = KeelRadius #full height of lens

LensOffset = 0. #* u.kilometer #horizontal offset of lens from center of box

SurfaceLayer = 300. # Thickness of surface-tracking particles in keel.

SurfaceShell = 5. # Depth of surface sampling (to avoid velocity BCs)


# In[11]:


#Density and Viscosity Variables: Porosity, Salt Content, Prescribed Density.

# Porosity of each model region.
PhiLens = 0.0
PhiCrust = 0.00
PhiShell = 0.0
PhiKeel = 0.0 #PhiCrust
#PhiAir = 0.5

# Salt fraction of each model region.

fSLens = 0.0
fSCrust = 0.0
fSShell = 0.0
fSKeel = 0.2

#Prescribed density
RBG = 0.0 #Density of non-keel regions
DRK = 1.0 #fraction of maximum density difference in keel

#Prescribed keel weakening
EtaFrac = 1000.0 #magnitude of viscous weakening in keel


# ## Model Domain Setup

# In[12]:


TopShell = 30000.# * u.kilometer


TS = TopShell / KL_meters#non_dimensionalise(TopShell)
CT = CrustThickness/ KL_meters#non_dimensionalise(CrustThickness)
KR = KeelRadius / KL_meters#non_dimensionalise(KeelRadius)
KW = KeelWidth / KL_meters
KW2 = KeelWidth2 / KL_meters
KD = KeelDepth / KL_meters
LG = LensGap / KL_meters#non_dimensionalise(LensGap)
LW = LensWidth / KL_meters#non_dimensionalise(LensWidth)
LS = LensHeight / KL_meters#non_dimensionalise(LensHeight)
LO = LensOffset / KL_meters#non_dimensionalise(LensOffset)
LH = LensHeight / KL_meters#non_dimensionalise(LensHeight)
SL = SurfaceLayer / KL_meters
SS = SurfaceShell / KL_meters

if CircleKeel == False:
    KR = KD
else:
    KD = KR
    KW = KR

#Coordinate setup based on above variable assignments
if Symmetry == True:
    Lx = Lx / 2.0
    CenterX = Lx
    Xres = int(Xres / 2)
else:
    CenterX = Lx / 2.0
    
CenterY = TS / 2.0

KeelEdge = CenterX - KW
KE = KeelEdge

KeelEdge2 = CenterX + KW2
KE2 = KeelEdge2

BottomCrust = TS - CT
BC = BottomCrust

BottomKeel = BC - KD
BK = BottomKeel

LensRadius = LH / 2.
LR = LensRadius

TopLens = BottomCrust - KR - LG
TL = TopLens

BottomLens = TopLens - LH
BL = BottomLens

LeftLens = CenterX - LW + LO
LL = LeftLens

RightLens = CenterX + LW + LO
RL = RightLens

LensCenter = TL - LR
LC = LensCenter


# In[13]:


#Mesh setup, definition of mesh initial conditions
mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"),
                                 elementRes  = (Xres,Yres),
                                 minCoord    = (0.0,0.0),
                                 maxCoord    = (Lx,Ly))

# Setup velocity and pressure field
velocityField       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount = 2 )
pressureField       = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount = 1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount = 1)
temperatureField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )

#material indexes
#AirIndex = 0
ShellIndex = 1
CrustIndex = 2
KeelIndex = 3
LensIndex = 4
SurfaceIndex = 5


# In[14]:


#Establish input/output path. I have this setup to run 4-8 models in a "batch", each with a "trial" number that outputs 
#Paraview results to its own folder. Make sure when saving Paraview files, you include a "/..." to the beginning
#of the file handle so that it generates a new folder.

outputPath = 'IceDripDataBatch'+str(Batch)+'/Trial'+str(Trial)

os.makedirs(outputPath, exist_ok=True)

#if not os.path.exists(outputPath):
#    os.makedirs(outputPath)
inputPath = outputPath


# In[15]:



velocityField.data[:]       = [0.,0.]


# In[16]:


if(LoadFromFile == True):
# set up mesh to import data--Useful if checkpointed data does not match model resolution
    data = np.loadtxt(inputPath+'IceDrip_Summary.out', unpack=True )
    timeData = np.loadtxt(inputPath+'ndTime.txt')
    #timeData = (timeData * (60 * 60 * 24 * 365 * 1000)) / Kt_seconds
    timeVal, vrmsVal = data[0].tolist(), data[1].tolist()
    step_prev = len(timeVal)
    try:
        time_start = timeData[-1]
    except:
        time_start = 0.0
    print ('Begining at t = ',time_start,' after having completed ',step_prev,' steps')
    
    temperatureField.load(inputPath+'/temperatureField'+str(Trial)+'.'+str(step_prev)+'.h5')
    temperatureDotField.load(inputPath+'/IceDrip_TDot.h5')
    pressureField.load(inputPath+'/IceDrip_P.h5')
    velocityField.load(inputPath+'/velocityField'+str(Trial)+'.'+str(step_prev)+'.h5')
    #Load material data onto swarm
    swarm = uw.swarm.Swarm( mesh=mesh )
    materialVariable = swarm.add_variable( dataType="int", count=1)
    swarm.load(outputPath+"/Swarm"+str(Trial)+"."+str(step_prev)+".h5")
    materialVariable.load(outputPath+"/MVar"+str(Trial)+"."+str(step_prev)+".h5")
    swarmStress = swarm.add_variable(dataType = "float", count=1)
    population_control = uw.swarm.PopulationControl(swarm,aggressive=True, particlesPerCell=20)
    xTracer = np.loadtxt(inputPath+'xTracer.txt')
    yTracer = np.loadtxt(inputPath+'yTracer.txt')
    xTracer_start = xTracer[-1] / 30.
    yTracer_start = yTracer[-1] / 30.
    print(xTracer_start)
    
    #Will also need to load stress history swarm variable if/when viscoelasticity is integrated
    
    # load summary statistics into arrays


# In[17]:


#material domain allocation-define polygons in model domain that then get assigned physical characteristics.

TopKeela = np.array([ (CenterX,TS), (CenterX,BC), (CenterX-KR, BC), (CenterX-KR, TS) ])
TopKeel = fn.shape.Polygon(TopKeela)

TaperKeela = np.array([ (CenterX, TS), (CenterX, BC - KD), (CenterX - KW, BC), (CenterX - KW, TS) ])
TaperKeelb = np.array([ (CenterX, TS), (CenterX, BC - KD), (KE2, BC), (KE2, TS) ])

SquareKeela = np.array([ (CenterX, TS), (CenterX, BC - KD), (CenterX - KW, BC - KD), (CenterX - KW, TS) ])
SquareKeelb = np.array([ (CenterX, TS), (CenterX, BC - KD), (KE2, BC - KD), (KE2, TS) ])

if Symmetry == True:
    if Taper == True:
        SquareKeel = fn.shape.Polygon(TaperKeela)
    else:
        SquareKeel = fn.shape.Polygon(SquareKeela)
    Surfacea = np.array([ (CenterX,TS), (CenterX,(TS-SL)), (CenterX-KW, (TS-SL)), (CenterX-KW, TS) ])
    Surface = fn.shape.Polygon(Surfacea)

else:
    if Taper == True:
        SquareKeel1 = fn.shape.Polygon(TaperKeela)
        SquareKeel2 = fn.shape.Polygon(TaperKeelb)
        SquareKeel = SquareKeel1 | SquareKeel2
    else:
        SquareKeel1 = fn.shape.Polygon(SquareKeela)
        SquareKeel2 = fn.shape.Polygon(SquareKeelb)
        SquareKeel = SquareKeel1 | SquareKeel2
    Surfacea = np.array([ (CenterX,TS), (CenterX,(TS-SL)), (CenterX-KW, (TS-SL)), (CenterX-KW, TS) ])
    Surfaceb = np.array([ (CenterX,TS), (CenterX,(TS-SL)), (KE2, (TS-SL)), (KE2, TS) ])
    Surface1 = fn.shape.Polygon(Surfacea)
    Surface2 = fn.shape.Polygon(Surfaceb)
    Surface = Surface1 | Surface2

Lensa = np.array([ (CenterX,TL), (LL,TL), (LL, BL), (CenterX, BL) ])
Lens = fn.shape.Polygon(Lensa)


# In[18]:


if(LoadFromFile == False):
#Initialize mesh variables and define initial temperature conditions. Set up particle swarm.
    velocityField.data[:]       = [0.,0.]
    pressureField.data[:]       = 0.
    temperatureDotField.data[:] = 0.
    #temperatureField.data[:] = 1.0
    timeVal = []
    vrmsVal = []
    swarm = uw.swarm.Swarm( mesh=mesh )
    materialVariable = swarm.add_variable( dataType="int", count=1)
    swarmStress = swarm.add_variable(dataType = "float", count=1)
    slabvar1 = swarm.add_variable( dataType="int", count=1)
    slabvar2 = swarm.add_variable( dataType="int", count=1)
    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm = swarm, particlesPerCell=20)
    swarm.populate_using_layout( layout=swarmLayout )
    population_control = uw.swarm.PopulationControl(swarm,aggressive=True, particlesPerCell=20)

    materialVariable.data[:] = ShellIndex
       
#    for index, coord in enumerate(mesh.data):
#        slabProf = -(Tc / slabThickness) * coord[1] + (Tc / slabThickness)
#        adiabat = -(slabThickness / Tc) * coord[1] + 1.
#        if coord[1] >= h:
#            temperatureField.data[index] = slabProf
#        if coord[1] < h:
#            temperatureField.data[index] = adiabat

            
    for index in range( len(swarm.particleCoordinates.data) ):
        coord = swarm.particleCoordinates.data[index][:]
        x = coord[0]
        z = coord[1]
        xx1 = x - CenterX
        zz1 = z - BC
        condition1 = (xx1*xx1 + zz1*zz1 < KR**2)
        xx2 = x - LL
        zz2 = z - LC
        condition2 = (xx2*xx2 + zz2*zz2 < LR**2)
        #if lowerMantle.evaluate(tuple(coord)):
            #materialVariable.data[index] = lowerMantleIndex
        if(coord[1] >= BC):
            materialVariable.data[index] = CrustIndex
        if(CircleKeel == True):
            if (condition1 == True):
                materialVariable.data[index] = KeelIndex
            elif(TopKeel.evaluate(tuple(coord))):
                materialVariable.data[index] = KeelIndex
        elif(CircleKeel == False):
            if SquareKeel.evaluate(tuple(coord)):
                materialVariable.data[index] = KeelIndex


        #if StickyAir.evaluate(tuple(coord)):
            #materialVariable.data[index] = AirIndex
        if(Surface.evaluate(tuple(coord))):
            materialVariable.data[index] = SurfaceIndex
        if(LensPresence == True): 
            if Lens.evaluate(tuple(coord)):
                materialVariable.data[index] = LensIndex
            if condition2 == True:
                materialVariable.data[index] = LensIndex


# In[19]:


#Tracer particle setup

# build a tracer swarm with one particle
tracerParticle1 = uw.swarm.Swarm(mesh)
advector_tracer1 = uw.systems.SwarmAdvector( swarm=tracerParticle1, velocityField=velocityField, order=2 )

tracerParticle2 = uw.swarm.Swarm(mesh)
advector_tracer2 = uw.systems.SwarmAdvector( swarm=tracerParticle2, velocityField=velocityField, order=2 )

tracerParticle3 = uw.swarm.Swarm(mesh)
advector_tracer3 = uw.systems.SwarmAdvector( swarm=tracerParticle3, velocityField=velocityField, order=2 )

tracerParticle4 = uw.swarm.Swarm(mesh)
advector_tracer4 = uw.systems.SwarmAdvector( swarm=tracerParticle4, velocityField=velocityField, order=2 )

tracerParticle5 = uw.swarm.Swarm(mesh)
advector_tracer5 = uw.systems.SwarmAdvector( swarm=tracerParticle5, velocityField=velocityField, order=2 )

#Setup temperature tracer particle. It has a variable "tracerTemp" that stores temperature data at its current location.
tracerParticle6 = uw.swarm.Swarm(mesh)
advector_tracer6 = uw.systems.SwarmAdvector( swarm=tracerParticle6, velocityField=velocityField, order=2 )
tracerVariable = tracerParticle6.add_variable( dataType="int", count=1)
tracerVariable.data[:] = 1
tracerTemp = tracerParticle6.add_variable( dataType="float", count=1 )

# build a numpy array with one particle, specifying its exact location
if CircleKeel == True:
    x_pos1 = CenterX - KR
    x_pos2 = CenterX - (KR/2)
    x_pos3 = CenterX
    x_pos4 = CenterX - (KR/4)
else:
    x_pos1 = CenterX - KW
    x_pos2 = CenterX - (KW/2)
    x_pos3 = CenterX
    x_pos4 = CenterX - (KW/4)
    

y_pos1 = TS
y_pos2 = BC
y_pos3 = BC-KR
y_pos4 = TS - SL
y_pos5 = TS - SS

#Tracer particles 1-5 currently only track either X coord or Y coord.

coord_array1 = np.array(object=(x_pos1,y_pos1),ndmin=2)
tracerParticle1.add_particles_with_coordinates(coord_array1)

if LoadFromFile == False:
    coord_array2 = np.array(object=(x_pos2,y_pos1),ndmin=2)
    tracerParticle2.add_particles_with_coordinates(coord_array2)
else:
    xTracer_start=xTracer_start+x_pos2
    coord_array2 = np.array(object=(xTracer_start,y_pos1),ndmin=2)
    tracerParticle2.add_particles_with_coordinates(coord_array2)
    
coord_array3 = np.array(object=(x_pos3,y_pos2),ndmin=2)
tracerParticle3.add_particles_with_coordinates(coord_array3)
coord_array4 = np.array(object=(x_pos3,y_pos3),ndmin=2)
tracerParticle4.add_particles_with_coordinates(coord_array4)

if LoadFromFile == False:
    coord_array5 = np.array(object=(x_pos3,y_pos4),ndmin=2)
    tracerParticle5.add_particles_with_coordinates(coord_array5)
else:
    coord_array5 = np.array(object=(x_pos3,yTracer_start),ndmin=2)
    tracerParticle5.add_particles_with_coordinates(coord_array5)

#Tracer Particle 6 tracks X coord, Y Coord, and Temperature.
#coord_array6 = np.array(object=(x_pos4,y_pos5),ndmin=2)
#tracerParticle6.add_particles_with_coordinates(coord_array6)

# define x-y coordinate `min_max` functions. As there is only one particle finding the min coord will return its coord.
fn_ycoord = fn.view.min_max( fn.coord()[1] )
fn_xcoord = fn.view.min_max( fn.coord()[0] )

def GetSwarm1XMin():
    fn_xcoord.reset()
    fn_xcoord.evaluate(tracerParticle1)
    return fn_xcoord.min_global()
def GetSwarm2XMin():
    fn_xcoord.reset()
    fn_xcoord.evaluate(tracerParticle2)
    return fn_xcoord.min_global()
def GetSwarm3YMin():
    fn_ycoord.reset()
    fn_ycoord.evaluate(tracerParticle3)
    return fn_ycoord.min_global()
def GetSwarm4YMin():
    fn_ycoord.reset()
    fn_ycoord.evaluate(tracerParticle4)
    return fn_ycoord.min_global()
def GetSwarm5YMin():
    fn_ycoord.reset()
    fn_ycoord.evaluate(tracerParticle5)
    return fn_ycoord.min_global()
#def GetSwarm6XMin():
#    fn_xcoord.reset()
#    fn_xcoord.evaluate(tracerParticle6)
#    return fn_xcoord.min_global()
#def GetSwarm6YMin():
#    fn_ycoord.reset()
#    fn_ycoord.evaluate(tracerParticle6)
#    return fn_ycoord.min_global()


# In[20]:


xmin1 = GetSwarm1XMin()
if rank==0:
    print('Minimum x value for keel surface boundary = {0:.3e}'.format(xmin1))
xmin2 = GetSwarm2XMin()
if rank==0:
    print('Minimum x value for keel surface interior = {0:.3e}'.format(xmin2))
ymin1 = GetSwarm3YMin()
if rank==0:
    print('Minimum y value for keel interior = {0:.3e}'.format(ymin1))
ymin2 = GetSwarm4YMin()
if rank==0:
    print('Minimum y value for keel base = {0:.3e}'.format(ymin2))
ymin3 = GetSwarm3YMin()
if rank==0:
    print('Minimum y value for keel base = {0:.3e}'.format(ymin3))
#xmin6 = GetSwarm6XMin()
#if rank==0:
#    print('Minimum x value for Tracer 6 = {0:.3e}'.format(xmin6))


# In[21]:


figParticle = glucifer.Figure(figsize=(960,300))
figParticle.append( glucifer.objects.Points(swarm, materialVariable, pointSize=2, colours='white green red purple blue') )
figParticle.append( glucifer.objects.Mesh(mesh, opacity=0.1))
figParticle.show()


# In[22]:


dTc = Tmin_s - Tc_s
dzc = CT
dzl = TS - BK
dTm = Tc_s - Tmax_s

#define initial temperature conditions according to set flags and variables defining drip nucleus shape.

if(LoadFromFile == False):
    for index, coord in enumerate(mesh.data):
        temperatureField.data[:] = 1.0
    for index, coord in enumerate(mesh.data):
        x = coord[0]
        z = coord[1]
        xx = x - CenterX
        zz = z - BC
        condition = (xx*xx + zz*zz < KR**2)
        dzlt = BK + ((BK-BC)/(CenterX-KE))*(coord[0]-CenterX)
        dzlt2 = BK + ((BC-BK)/(KE2-CenterX))*(coord[0]-CenterX)
        if CircleKeel == True:  
            if(condition == True):
                temperatureField.data[index] = Tmin_s + (dTc/dzl)*(coord[1]-TS)
            elif coord[1] <= BC:
                temperatureField.data[index] = (dTm/BC)*coord[1]+Tmax_s
            elif coord[1] >= BC:
                temperatureField.data[index] = Tmin_s + (dTc/dzc)*(coord[1]-TS)
        elif CenterX >= coord[0] >= KE:
            if Taper == True:
                if coord[1] >= dzlt:
                    temperatureField.data[index] = Tmin_s + (dTc/(TS - dzlt))*(coord[1]-TS)
                if coord[1] <= dzlt:
                    temperatureField.data[index] = (dTm/BC)*coord[1]+Tmax_s
            elif coord[1] >= BK:
                temperatureField.data[index] = Tmin_s + (dTc/dzl)*(coord[1]-TS)
            if coord[1] <= BK:
                temperatureField.data[index] = (dTm/BC)*coord[1]+Tmax_s
        elif KE2 >= coord[0] >= CenterX:
            if Taper == True:
                if coord[1] >= dzlt2:
                    temperatureField.data[index] = Tmin_s + (dTc/(TS - dzlt2))*(coord[1]-TS)
                if coord[1] <= dzlt2:
                    temperatureField.data[index] = (dTm/BC)*coord[1]+Tmax_s
            elif coord[1] >= BK:
                temperatureField.data[index] = Tmin_s + (dTc/dzl)*(coord[1]-TS)
            if coord[1] <= BK:
                temperatureField.data[index] = (dTm/BC)*coord[1]+Tmax_s
        elif coord[1] <= BC:
            temperatureField.data[index] = (dTm/BC)*coord[1]+Tmax_s
        elif coord[1] >= BC:
            temperatureField.data[index] = Tmin_s + (dTc/dzc)*(coord[1]-TS)
        if coord[1] >= TS:
            temperatureField.data[index] = Tmin_s
        


# In[23]:


viz_opts = { "figsize"     : (2400,1200),
             "boundingBox" : ( (-3.0, 0.0, 0.0), ( 3.0, 1.0, 0.0) ),
             "edgecolour"  :  "black",
             "quality"     :  3,          # antialiasing
             "align"       : "right",     # colour bar alignment
             "size"        : (0.83,0.01), # colour bar size
             "position"    : 0. }         # colour bar position

fig1 = glucifer.Figure(**viz_opts)
fig1.append( glucifer.objects.Surface(mesh, 100.+(173.*temperatureField), colours = "#19ff00 #00c6ff") )
fig1.append( glucifer.objects.VectorArrows(mesh, velocityField / 500., arrowHead=0.2, scaling=0.1) )
#fig1.append( glucifer.objects.Mesh(mesh))
fig1.show()
fig1.save_image("tfig.png")

#coord6 = [GetSwarm6XMin(), GetSwarm6YMin()]
#print(coord6)
#temp1 = temperatureField.evaluate(tracerParticle6)
#temp2 = temperatureField.evaluate(coord6)
#print(temp1)
#print(temp2)

#This command tells the model to sample temperature at tracerParticle6's current location and store it in the mesh variable
#we defined earlier.

#tracerTemp = temperatureField.evaluate(tracerParticle6)
#print(tracerTemp)


# In[24]:


#This cell defines density perturbations allocated by material assignment.

PhiMap = { LensIndex  : PhiLens,
           CrustIndex : PhiCrust,
           ShellIndex : PhiShell,
           KeelIndex  : PhiKeel,
           SurfaceIndex : PhiKeel}
           #AirIndex   : PhiAir   }

fSMap = {  LensIndex    : fSLens,
           CrustIndex   : fSCrust,
           ShellIndex   : fSShell,
           KeelIndex    : fSKeel,
           SurfaceIndex : fSKeel}
           #AirIndex   : fSAir   }

Phi = fn.branching.map( fn_key = materialVariable, mapping = PhiMap)

fS = fn.branching.map( fn_key = materialVariable, mapping = fSMap)

figPhi = glucifer.Figure(figsize=(960,300))
figPhi.append( glucifer.objects.Points(swarm, Phi, pointSize=2,) )
figPhi.append( glucifer.objects.Mesh(mesh, opacity=0.1))
#figPhi.show()

figfS = glucifer.Figure(figsize=(960,300))
figfS.append( glucifer.objects.Points(swarm, fS, pointSize=2,) )
figfS.append( glucifer.objects.Mesh(mesh, opacity=0.1))
#figfS.show()


# In[25]:


#Defining Density Function
#Rb = Ra

if SaltDensity == True:
    DeltaRho = -rho_s*Phi+fS*(rhoSalt_s-rho_s)

    DeltaRho = fn.misc.min(DeltaRho,DelRho_s) # This ensures the ice is never more dense than water.

    Density = DeltaRho #+ rho_s

    DensityMap = { LensIndex  : Density,
                   CrustIndex : Density,
                   ShellIndex : Density,
                   KeelIndex  : Density,
                   SurfaceIndex  : Density }#DeltaRho  }
                   #AirIndex   : DeltaRho   }
else:
    rhoBG = RBG
    DeltaRhoKeel = DelRho_s * DRK # Max density difference is DelRho_s (0.89). Scaled to density of ice. positive is more dense.
                                  # Manipulate number in denominator to modulate density difference of keel.
    DensityMap = { LensIndex  : rhoBG,
                   CrustIndex : rhoBG,
                   ShellIndex : rhoBG,
                   KeelIndex  : DeltaRhoKeel,
                   SurfaceIndex  : DeltaRhoKeel}

DensityFn = fn.branching.map( fn_key = materialVariable, mapping = DensityMap)


DensityFn_s = Ra * temperatureField - Rb * (DensityFn / DelRho_s)

figDensity = glucifer.Figure(figsize=(960,300))
figDensity.append( glucifer.objects.Points(swarm, DensityFn, pointSize=2,) )
figDensity.append( glucifer.objects.Mesh(mesh, opacity=0.1))
#figDensity.show()

figDensity_s = glucifer.Figure(figsize=(960,300))
figDensity_s.append( glucifer.objects.Points(swarm, DensityFn_s, pointSize=2,) )
figDensity_s.append( glucifer.objects.Mesh(mesh, opacity=0.1))
#figDensity_s.show()


# In[26]:


#define rheology

etamax_s = etamax_s #Testing viscous flow law stuff

#Temperature-Dependent viscous rheology
frankKam = (etamax_s * fn.math.exp(-1.0 * np.log(etamax_s) * temperatureField)) #* ((1. - Phi)**Beta)


#Plastic failure criterion. Currently not active.
strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))

cohesion = sigy_s #2449479.5 

#vonMises = 1000.
    
vonMises = 0.5 * cohesion / (strainRate_2ndInvariant+1.0e-18)

figStrainRate = glucifer.Figure(name="strain Rate", figsize=(960,300) )
StrainRate =glucifer.objects.Surface(mesh, strainRate_2ndInvariant)
figStrainRate.append( StrainRate )
StrainRate.colourBar["ticks"] = 10
#figStrainRate.show()


# In[27]:


ShellVisc = frankKam
CrustVisc = fn.exception.SafeMaths(fn.misc.min(vonMises, ShellVisc))
KeelVisc1 = frankKam / EtaFrac # Viscous-only Keel. Manipulate number in denominator to weaken keel wrt rest of crust.
KeelVisc2 = fn.exception.SafeMaths(fn.misc.min(vonMises, KeelVisc1)) # adds plastic failure
StickyAir = 1.0

viscousOnlyMap = { LensIndex  : ShellVisc,
                   CrustIndex : ShellVisc,
                   ShellIndex : ShellVisc,
                   KeelIndex  : KeelVisc1,
                   SurfaceIndex  : KeelVisc1}
                 #AirIndex   : StickyAir   }
    
viscosityMap = {  LensIndex  : ShellVisc,
                  CrustIndex : CrustVisc,
                  ShellIndex : ShellVisc,
                  KeelIndex  : KeelVisc2,
                  SurfaceIndex  : KeelVisc2}

viscousOnlyFn = fn.branching.map( fn_key = materialVariable, mapping = viscousOnlyMap)
viscosityFn = fn.branching.map( fn_key = materialVariable, mapping = viscosityMap)

figViscosity= glucifer.Figure(name="Viscosity", figsize=(960,300) )
figViscosity.append( glucifer.objects.Mesh(mesh, opacity=0.1))
figViscosity.append( glucifer.objects.Points(swarm     = swarm, 
                                             fn_colour = viscosityFn, 
                                             pointSize = 1.0, 
                                             #fn_mask   = materialFilter,
                                             logScale  = True))

stressInv = 2. * viscousOnlyFn * (strainRate_2ndInvariant + 1.0e-18)

swarmStress.data[:] = stressInv.evaluate(swarm) * K_pascal

maxStress = fn.view.min_max(swarmStress)

def GetMaxStress():
    maxStress.reset()
    maxStress.evaluate(swarm)
    return maxStress.max_global()
    

#figViscosity.colourbar["ticks"] = 6
figViscosity.show()

#figStress = vis.Figure( store, figsize=(960,300) )
#figStress.append( vis.objects.Points(swarm, 2.0*viscosityMapFn*strainRate_2ndInvariant, pointSize=2, logScale=True) )


# In[28]:


z_hat = (0.0, 1.0)

buoyancyFn = DensityFn_s * z_hat


# In[29]:


#Define T and V BCs
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
topWall = mesh.specialSets["MaxJ_VertexSet"]
bottomWall = mesh.specialSets["MinJ_VertexSet"]

    
TBC = jWalls #iWalls + jWalls
VxBC = iWalls
VyBC = jWalls

velocityBCs = uw.conditions.DirichletCondition( variable        = velocityField,
                                                indexSetsPerDof = (VxBC, VyBC) )

temperatureBCs = uw.conditions.DirichletCondition( variable        = temperatureField,
                                                   indexSetsPerDof = (TBC) )


# ## System and Solver Setup, Time-Stepping Loop

# In[30]:


#Systems setup
stokes = uw.systems.Stokes( velocityField = velocityField,
                            pressureField = pressureField,
                            voronoi_swarm = swarm,
                            conditions    = velocityBCs,
                            fn_viscosity  = viscosityFn,
                            fn_bodyforce  = buoyancyFn )

stokesInit = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = velocityBCs,
                                fn_viscosity  = viscousOnlyFn,
                                fn_bodyforce  = buoyancyFn )

advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField,
                                         phiDotField    = temperatureDotField,
                                         velocityField  = velocityField,
                                         fn_diffusivity = 1.0,
                                         conditions     = temperatureBCs )

#These are mobile tracers, so we need to set up advectors for each one. If we want static tracers (such as for temperature) 
#don't make an advector.

advector1 = uw.systems.SwarmAdvector( swarm=swarm,
                                      velocityField = velocityField,
                                      order = 2)
advector2 = uw.systems.SwarmAdvector( swarm=tracerParticle1,
                                      velocityField = velocityField,
                                      order = 2)
advector3 = uw.systems.SwarmAdvector( swarm=tracerParticle2,
                                      velocityField = velocityField,
                                      order = 2)
advector4 = uw.systems.SwarmAdvector( swarm=tracerParticle3,
                                      velocityField = velocityField,
                                      order = 2)
advector5 = uw.systems.SwarmAdvector( swarm=tracerParticle4,
                                      velocityField = velocityField,
                                      order = 2)
advector6 = uw.systems.SwarmAdvector( swarm=tracerParticle5,
                                      velocityField = velocityField,
                                      order = 2)
#advector7 = uw.systems.SwarmAdvector( swarm=tracerParticle6,
#                                      velocityField = velocityField,
#                                      order = 2)
solverInit = uw.systems.Solver( stokesInit )
solver = uw.systems.Solver( stokes )


# In[31]:


#analysis toools setup

#vrms
intVdotV = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
vrms = math.sqrt( intVdotV.evaluate()[0] )


# In[32]:


#time stepping loop. Make sure to integrate all tracer advectors.
def update():
    dt1 = advDiff.get_max_dt()
    advDiff.integrate(dt1)
    advector1.integrate(dt1)
    advector2.integrate(dt1)
    advector3.integrate(dt1)
    advector4.integrate(dt1)
    advector5.integrate(dt1)
    advector6.integrate(dt1)
    return time+dt1, step+1

#initialize and solve
step_prev = len(timeVal)
time = 0.
step = step_prev
steps_end = 1
step_out = 1

# Set time to zero, unless we are loading from file.
try:
    time_start = timeData[-1]
except:
    time_start = 0.0
print ('Begining at t = ',time_start,' after having completed ',step_prev,' steps')


# In[33]:


if (Testing == True):
    nsteps=10
    step_out = 2
else:
    if Long == True:
        nsteps = 5000 #2500
    else:
        nsteps = 2500
    step_out = 500 #This defines how often the model generates output figures and files.
try:
    time_start = timeData[-1]
except:
    time_start = 0.0
print ('Begining at t = ',time_start,' after having completed ',step_prev,' steps')

step_prev = len(timeVal)
time = time_start
#Set up tracer arrays. These arrays track model time, x and y position for the tracers, % shortening, and temperature.
tTracers = np.zeros(nsteps)
xTracer1 = np.zeros(nsteps)
xTracer2 = np.zeros(nsteps)
yTracer3 = np.zeros(nsteps)
yTracer4 = np.zeros(nsteps)
yTracer5 = np.zeros(nsteps)
xTracer6 = np.zeros(nsteps)
yTracer7 = np.zeros(nsteps)
ndTime = np.zeros(nsteps)
Shortening1 = np.zeros(nsteps)
Shortening2 = np.zeros(nsteps)
#tempTracer = np.zeros(nsteps)
Compression2 = np.zeros(nsteps)
stressMax = np.zeros(nsteps)

if LoadFromFile == False:
    step = 0
    step_prev = 0
else:
    step = step_prev
    nsteps = nsteps + step_prev

# Save initial h5 files of the mesh and the swarm. If your mesh is static, this is the only mesh file you will need.
meshFileHandle = mesh.save(outputPath+"/Mesh.h5")
swarmFileHandle = swarm.save(outputPath+"/Swarm.h5")
#tracerFileHandle = tracerParticle6.save(outputPath+"/Tracer6.h5")

if LoadFromFile == True:
    solverInit.solve(nonLinearIterate = True)
    time, step = update()
    population_control.repopulate()
    nsteps = nsteps + 1
    step_prev = step_prev + 1

while step < nsteps:
    solverInit.solve(nonLinearIterate=True)
    vrms = math.sqrt( intVdotV.evaluate()[0] )
    vrmsVal.append(vrms)
    #at every time step, read position of each tracer coordinate we're interested in. Also, use the command from earlier
    #to read the current temperature of our temp particle.
    swarmStress.data[:] = stressInv.evaluate(swarm) * K_pascal
    xmin1 = GetSwarm1XMin()
    xmin2 = GetSwarm2XMin()
    #xmin3 = GetSwarm6XMin()
    ymin1 = GetSwarm3YMin()
    ymin2 = GetSwarm4YMin()
    ymin3 = GetSwarm5YMin()
    #ymin4 = GetSwarm6YMin()
    stressM = GetMaxStress()
    #tracerTemp = temperatureField.evaluate(tracerParticle6)
    
    
    if rank==0:
        #record re-scaled positional data, etc.
        step2 = step - step_prev
        xTracer1[step2] = xmin1 * 30.
        Shortening1[step2] = ((CenterX-x_pos1)-(CenterX-xmin1))/(CenterX-x_pos1)
        xTracer2[step2] = xmin2 * 30.
        Shortening2[step2] = ((CenterX-x_pos2)-(CenterX-xmin2))/(CenterX-x_pos2)
        Compression2[step2] = xTracer2[step2] - xTracer2[0]
        yTracer3[step2] = ymin1 * 30.
        yTracer4[step2] = ymin2 * 30.
        yTracer5[step2] = ymin3 * 30.
        #xTracer6[step2] = xmin3 * 30.
        #yTracer7[step2] = ymin4 * 30.
        tTracers[step2] = (time * Kt_seconds) / (60 * 60 * 24 * 365 * 1000)
        ndTime[step2] = time
        stressMax[step2] = (stressM * K_pascal) / 1e6
        #tempTracer[step2] = (tracerTemp * KT_degrees) + 100.
        

        
    if step % step_out == 0: #This chunk of the step loop triggers every 50 steps.
        
        #Every time you generate a new swarm-based xdmf, you must save a new swarm .h5 to account for its advection.
        #When viewing files in paraview, it will automatically group sequential files and arrange them by number.
        #In order for Paraview to open an xdmf file, the parent .h5 file AND the base mesh or swarm file MUST be in
        #The same folder. Additionally, this folder must contain ONLY the results of a single model run.
        
        #When defining Paraview file handles, we want to separate them by timestep as shown below.
        #This swarm file handle will translate to "/Swarm5.50.h5", for example, where ".50." represents the timestep.
        swarmFileHandle = swarm.save(outputPath+"/Swarm"+str(Trial)+"."+str(step)+".h5")
        #tracerFileHandle = tracerParticle6.save(outputPath+"/Tracer6T"+str(Trial)+"."+str(step)+".h5")
        temperatureField.save(outputPath+'/IceDrip_T.out')
        temperatureDotField.save(outputPath+'/IceDrip_TDot.out')
        velocityField.save(outputPath+'/IceDrip_v.out')
        pressureField.save(outputPath+'/IceDrip_P.out')

        #Below are the xdmf generation commands. First, we define a file handle, then we save a base .h5 file based
        #on that file handle. Then we use the ".xdmf" command to identify what aspect of the model its recording and whether
        #it uses the mesh or the swarm as a base.
        
        
        
        velocityFile = outputPath+"/velocityField"+str(Trial)+"."+str(step)
        vFH      = velocityField.save(velocityFile+".h5")
        velocityField.xdmf( velocityFile, vFH, "velocity", meshFileHandle, "TheMesh", time )
        
        pressureFile = outputPath+"/pressureField"+str(Trial)+"."+str(step)
        pFH      = pressureField.save(pressureFile+".h5")
        pressureField.xdmf(pressureFile, pFH, "pressure", meshFileHandle, "TheMesh", time )
        
        temperatureFile = outputPath+"/temperatureField"+str(Trial)+"."+str(step)
        TFH      = temperatureField.save(temperatureFile+".h5")
        temperatureField.xdmf(temperatureFile, TFH, "temperature", meshFileHandle, "TheMesh", time )
        
        #filename = outputPath+"/Particles{}.".format(Trial)+str(step) 
        #swFH      = swarm.save(filename+".h5")
        #pressureField.xdmf(filename, swFH, "swarm", swarmFileHandle, "swarm", time )
        
        particleFile = outputPath+"/MVar"+str(Trial)+"."+str(step) 
        MVFH      = materialVariable.save(particleFile+".h5")
        materialVariable.xdmf(particleFile, MVFH, "materialVariable", swarmFileHandle, "TheSwarm", time )
        
        stressFile = outputPath+"/Stress"+str(Trial)+"."+str(step) 
        STFH      = swarmStress.save(stressFile+".h5")
        swarmStress.xdmf(stressFile, STFH, "swarmStress", swarmFileHandle, "TheSwarm", time )
     
            
    time, step = update()
    population_control.repopulate()
    timeVal.append(step)
    if step % step_out == 0:
        
        np.savetxt( outputPath+'IceDrip_Summary.out', np.c_[timeVal, vrmsVal])
        vrms = stokes.velocity_rms()
        print('step = {0:6d}; time = {1:.3e}; Vrms = {2:.3e}; xmin1 = {3:.3e}; ymin2 = {4:.3e}'.format(step,time,vrms,xmin1,ymin2))
        
        swarmFileHandle = swarm.save(outputPath+"/Swarm"+str(Trial)+"."+str(step)+".h5")
        #tracerFileHandle = tracerParticle6.save(outputPath+"/Tracer6T"+str(Trial)+"."+str(step)+".h5")
        temperatureField.save(outputPath+'/IceDrip_T.out')
        temperatureDotField.save(outputPath+'/IceDrip_TDot.out')
        temperatureDotField.save(outputPath+'/IceDrip_TDot.h5')
        velocityField.save(outputPath+'/IceDrip_v.out')
        pressureField.save(outputPath+'/IceDrip_P.out')
        pressureField.save(outputPath+'/IceDrip_P.h5')

        #Below are the xdmf generation commands. First, we define a file handle, then we save a base .h5 file based
        #on that file handle. Then we use the ".xdmf" command to identify what aspect of the model its recording and whether
        #it uses the mesh or the swarm as a base.
        
        
        
        velocityFile = outputPath+"/velocityField"+str(Trial)+"."+str(step)
        vFH      = velocityField.save(velocityFile+".h5")
        velocityField.xdmf( velocityFile, vFH, "velocity", meshFileHandle, "TheMesh", time )
        
        pressureFile = outputPath+"/pressureField"+str(Trial)+"."+str(step)
        pFH      = pressureField.save(pressureFile+".h5")
        pressureField.xdmf(pressureFile, pFH, "pressure", meshFileHandle, "TheMesh", time )
        
        temperatureFile = outputPath+"/temperatureField"+str(Trial)+"."+str(step)
        TFH      = temperatureField.save(temperatureFile+".h5")
        temperatureField.xdmf(temperatureFile, TFH, "temperature", meshFileHandle, "TheMesh", time )
        
        #filename = outputPath+"/Particles{}.".format(Trial)+str(step) 
        #swFH      = swarm.save(filename+".h5")
        #pressureField.xdmf(filename, swFH, "swarm", swarmFileHandle, "swarm", time )
        
        particleFile = outputPath+"/MVar"+str(Trial)+"."+str(step) 
        MVFH      = materialVariable.save(particleFile+".h5")
        materialVariable.xdmf(particleFile, MVFH, "materialVariable", swarmFileHandle, "TheSwarm", time )
        
        stressFile = outputPath+"/Stress"+str(Trial)+"."+str(step) 
        STFH      = swarmStress.save(stressFile+".h5")
        swarmStress.xdmf(stressFile, STFH, "swarmStress", swarmFileHandle, "TheSwarm", time )


# ## Post-Run Analysis

# In[34]:


#figDensity.show()

#figStrainRate.show()

#uw.utils.matplotlib_inline()


import matplotlib.pyplot as pyplot
fig4 = pyplot.figure()
fig4.set_size_inches(12, 6)
ax= fig4.add_subplot(1,1,1)
ax.plot(timeVal, vrmsVal, color = 'black')
ax.set_title('VRMS: Yres = %i' %Yres)
ax.set_xlabel('Timestep')
ax.set_ylabel('RMS velocity')
ax.set_xlim([1.0,nsteps])
fig4.savefig(outputPath+'vrms{}.png'.format(Trial))

#print(xTracer6)
#print(yTracer7)
#print(tempTracer)


# In[35]:


#Here we are recording tracer data to a text file.
if rank==0:
    nsteps = nsteps - step_prev
    fw = open(outputPath + "TracerInfo.txt","a")
    fw.write('Initial position: t = {0:.3f}, x1 = {1:.3f}, x2 = {2:.3f}, y1 = {3:.3f}, y2 = {4:.3f}, y3 = {5:.3f} \n'.format(tTracers[0], xTracer1[0], xTracer2[0], yTracer3[0], yTracer4[0], yTracer5[0]))
    fw.write('Final position:   t = {0:.3f}, x1 = {1:.3f}, x2 = {2:.3f}, y1 = {3:.3f}, y2 = {4:.3f}, y3 = {5:.3f} \n'.format(tTracers[nsteps-1], xTracer1[nsteps-1], xTracer2[nsteps-1], yTracer3[nsteps-1], yTracer4[nsteps-1], yTracer5[nsteps-1]))
    fw.write('Surface movement (km): x1 = {0:.3f}, x2 = {1:.3f}, y3 = {2:.3f} \n'.format(abs(xTracer1[0]-xTracer1[nsteps-1]), abs(xTracer2[0]-xTracer2[nsteps-1]), abs(yTracer5[0]-yTracer5[nsteps-1])))
    fw.write('Final Surface Shortening (%): x1 = {0:.3f}, x2 = {1:.3f}'.format(Shortening1[nsteps-1]* 100., Shortening2[nsteps-1] * 100.))
    fw.close()
    
    if LoadFromFile == False:
        np.savetxt(outputPath + "TimeTracer.txt", tTracers)
        np.savetxt(outputPath + "yTracer.txt", yTracer5)
        np.savetxt(outputPath + "xTracer.txt", Compression2)
        np.savetxt(outputPath + "x2Shortening.txt", Shortening2)
        np.savetxt(outputPath + "Stress.txt", stressMax)
        np.savetxt(outputPath + "ndTime.txt", ndTime)
        np.savetxt(outputPath + "VrmsVal.txt", vrmsVal)
    else:
        AlltTracers = np.loadtxt(outputPath + "TimeTracer.txt")
        AlltTracers = np.append(AlltTracers, tTracers)
        np.savetxt(outputPath + "TimeTracer.txt", AlltTracers)
        
        AllyTracers = np.loadtxt(outputPath + "yTracer.txt")
        AllyTracers = np.append(AllyTracers, yTracer5)
        np.savetxt(outputPath + "yTracer.txt", AllyTracers)
        
        AllxTracers = np.loadtxt(outputPath + "xTracer.txt")
        AllxTracers = np.append(AllxTracers, Compression2)
        np.savetxt(outputPath + "xTracer.txt", AllxTracers)
        
        AllShortening = np.loadtxt(outputPath + "x2Shortening.txt")
        AllShortening = np.append(AllShortening, Shortening2)
        np.savetxt(outputPath + "x2Shortening.txt", AllShortening)
        
        AllSTracers = np.loadtxt(outputPath + "Stress.txt")
        AllSTracers = np.append(AllSTracers, stressMax)
        np.savetxt(outputPath + "Stress.txt", AllSTracers)
        
        AllndTracers = np.loadtxt(outputPath + "ndTime.txt")
        AllndTracers = np.append(AllndTracers, ndTime)
        np.savetxt(outputPath + "ndTime.txt", AllndTracers)

    
    #uw.utils.matplotlib_inline()
    figTracer1 = pyplot.figure()
    figTracer1.set_size_inches(18, 6)
    ax1 = figTracer1.add_subplot(1,2,1)
    ax1.plot(tTracers, Shortening1, label = "x1")
    ax1.plot(tTracers, Shortening2, label = "x2")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Surface Shortening')
    ax1.legend()
    ax2 = figTracer1.add_subplot(1,2,2)
    ax2.plot(tTracers, yTracer3, color = "purple", label = "y1")
    ax2.plot(tTracers, yTracer4, color = "green", label = "y2")
    ax2.plot(tTracers, yTracer5, color = "red", label = "y3")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Keel Y Position')
    ax2.legend()
    figTracer1.savefig(outputPath+'Fig1Tracer{}.png'.format(Trial))
    
    #figTracer2 = pyplot.figure()
    #figTracer2.set_size_inches(18, 6)
    #ax3 = figTracer2.add_subplot(1,2,1)
    #ax3.plot(tTracers,stressMax, label = "Maximum Stress")
    #ax3.set_xlabel('Time')
    #ax3.set_ylabel('Stress (MPa)')
    #ax4 = figTracer2.add_subplot(1,2,2)
    #ax4.plot(tTracers, tempTracer, label = "Temperature Tracer")
    #ax4.set_xlabel('Time')
    #ax4.set_ylabel('Temperature')
    #figTracer2.savefig(outputPath+'Fig2Tracer{}.png'.format(Trial))

    


# In[36]:


np.savetxt(outputPath + "IceDripData{}.txt".format(Trial), (tTracers, xTracer1, xTracer2, yTracer3, yTracer4), fmt='%1.4e')


# In[37]:


figParticle.show()


# In[ ]:




