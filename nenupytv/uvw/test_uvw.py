import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt

# Convertor from degrees to radians.
dc = np.pi/180.0;

# xyz : local geographical coordinates, with x->East, y->North, z->Zenith.
# XYZ : Earth/Celestial coordinate system (located at the array location) 
#          with X->East, Z->NCP and Y to make a RH system.
# uvw : When HA=0, DEC=90, uvw is coincident with XYZ.  
#          v and w and NCP are always on a Great-circle, u completes the RH system
#          When 'v' or 'w' is the NCP, then 'u' points East.  

## Rotate counter-clockwise about x
def rotx( xyz , angle ):
    newxyz = np.array( [0.0,0.0,0.0], 'float' )
    ang = angle*dc
    newxyz[0] = xyz[0]
    newxyz[1] = xyz[1] * np.cos(ang) - xyz[2] * np.sin(ang)
    newxyz[2] = xyz[1] * np.sin(ang) + xyz[2] * np.cos(ang)
    return newxyz

## Rotate counter-clockwise about y
def roty( xyz , angle ):
    newxyz = np.array( [0.0,0.0,0.0], 'float' )
    ang = angle*dc
    newxyz[0] = xyz[0] * np.cos(ang) + xyz[2] * np.sin(ang)
    newxyz[1] = xyz[1]
    newxyz[2] = -1* xyz[0] * np.sin(ang) + xyz[2] * np.cos(ang)
    return newxyz

## Rotate counter-clockwise about z
def rotz( xyz , angle ):
    newxyz = np.array( [0.0,0.0,0.0], 'float' )
    ang = angle*dc
    newxyz[0] = xyz[0] * np.cos(ang) - xyz[1] * np.sin(ang)
    newxyz[1] = xyz[0] * np.sin(ang) + xyz[1] * np.cos(ang)
    newxyz[2] = xyz[2]
    return newxyz

## Three rotations. 
## Start with uvw aligned with local xyz
## Rotate about x by 90 deg, to get 'w' to point HA=0, DEC=0
## Rotate about x by -DEC 
## Rotate about z by -HA
def xyz2uvw( xyz , ha, dec ):
    newuvw = rotz( rotx( rotx( xyz, 90 ) , -1*dec ) , -1*ha*15 )
    return newuvw

def localxyz2uvw( xyz, hourangle, declination, latrot ):
    uvwdir = xyz2uvw( xyz, hourangle, declination )
    uvwdir = rotx( uvwdir, latrot )
    return uvwdir


######################################
## Antennas
# ID   Name  Station   Diam.    Long.         Lat.                Offset from array center (m)                ITRF Geocentric coordinates (m)        
#                                                                                            East         North           Elevation               x                    y                        z
# 0    ea06  N06       25.0 m   -107.37.06.9  +33.54.10.3        -54.0649      263.8778       -4.2273 -1601162.591000 -5041828.999000  3555095.896400
# 1    ea07  E05       25.0 m   -107.36.58.4  +33.53.58.8        164.9788      -92.8032       -2.5268 -1601014.462000 -5042086.252000  3554800.799800
# 2    ea11  E04       25.0 m   -107.37.00.8  +33.53.59.7        102.8054      -63.7682       -2.6414 -1601068.790300 -5042051.910200  3554824.835300
# 3    ea18  N09       25.0 m   -107.37.07.8  +33.54.19.0        -77.4346      530.6273       -5.5859 -1601139.485100 -5041679.036800  3555316.533200
######################################

## Calculate and draw all the coordinate systems.
def cdraw(hourangle=-3.5, declination=+20.0, obslatitude=34.0,antennalist=''):
    """
    hourangle : HA in hours. Zero is on the local meridian. -6.0 is rising, and +6.0 is setting
    declination : DEC in degrees. 0 is on celestial equator (XY plane), +90 is at NCP.
    obslatitude : Latitude of the Observatory ( VLA : 34.0 )
    antennalist : text file name containing the listobs output for antenna info (without headers)
    """
    
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure(num=1, figsize=(12,8))
    plt.clf()
    ax = fig.gca(projection='3d')
    
    #plt.clf()

    ## Length in meters of the axis lines.... ('radius of celestial sphere')
    AL = 700.0
    
    ## Draw the NSWE axes
    Xx = [-AL,+AL]
    Xy = [0,0]
    Xz = [0,0]
    Yx = [0,0]
    Yy = [-AL,+AL]
    Yz = [0,0]
    Zx = [0,0]
    Zy = [0,0]
    Zz = [0,1.5*AL]
    ax.plot(Xx,Xy,Xz, 'k')
    ax.plot(Yx,Yy,Yz, 'k')
    ax.plot(Zx,Zy,Zz, 'k')
    #ax.text( Xx[1], Xy[1], Xz[1], 'E,x' )
    #ax.text( Yx[1], Yy[1], Yz[1], 'N,y' )
    ax.text( Xx[1], Xy[1], Xz[1], 'E' )
    ax.text( Yx[1], Yy[1], Yz[1], 'N' )
    ax.text( Xx[0], Xy[0], Xz[0], 'W' )
    ax.text( Yx[0], Yy[0], Yz[0], 'S' )
    ax.text( Zx[1], Zy[1], Zz[1], 'zenith' )

    ##ax.add_patch(Rectangle( (-500, -500), 1000, 1000, alpha=0.1 )) 

    ## Use the antenna coordinates on the surface of the Earth, referenced to the array center (listobs output)

    if antennalist=='':
        eastlocs= np.array( [-54.0649,164.9788,102.8054,-77.4346] , 'float' )
        northlocs = np.array( [263.8778,-92.8032, -63.7682,530.6273] , 'float' )
        elevlocs = np.array( [-4.2273, -2.5268, -2.6414,-5.5859], 'float' )
        antnames = ['ea06','ea07','ea11','ea18']
    else:
        eastlocs, northlocs, elevlocs, antnames = readAntListFile(antennalist)

    ## Assign x->East and x-North. This is the local geographic csys
    Xlocs = eastlocs
    Ylocs = northlocs 
    
    ## Plot the antennas
    ax.plot( Xlocs, Ylocs,elevlocs+20,  'o', color='yellowgreen' )
    for ant in range(0,len(Xlocs)):
        ax.text( Xlocs[ant], Ylocs[ant], elevlocs[ant]+20, antnames[ant] , fontsize=8)

    ################################ Construct XYZ
    ## Start with  local xyz
    xdir1 = np.array( [AL,0,0], 'float' )
    ydir1 = np.array( [0,AL,0], 'float' )
    zdir1 = np.array( [0,0,AL], 'float' )

    ## Rotate by observatory latitude to get 'Z' to the NCP
    latrot = -90+obslatitude
    xdir = rotx( xdir1, latrot )
    ydir = rotx( ydir1, latrot )
    zdir = rotx( zdir1, latrot )
    ################################

    ################################ Construct uvw
    ## Start with local xyz
    ## Rotate to get 'z' pointed to where 'w' should be when HA=0, DEC=0
    ## Rotate by HA and DEC in appropriate directions.
    ## Rotate by observatory latitude.
    udir = localxyz2uvw( xdir1, hourangle, declination, latrot )
    vdir = localxyz2uvw( ydir1, hourangle, declination, latrot )
    wdir = localxyz2uvw( zdir1, hourangle, declination, latrot )
    
    ################################

    ## Define an origin ( for example, reference antenna )
    origin = np.array( [Xlocs[0], Ylocs[0], elevlocs[0]], 'float' )
    #origin = np.array( [0.0,0.0,0.0], 'float' )
    
    ################################
    ## Calculate UVWs for all antennas.
    axyz=np.array( [0.0,0.0,0.0], 'float')
    antuvws = np.zeros( (len(Xlocs), 3) ,'float')
    for ant in range(0,len(Xlocs)):
        axyz[0] = Xlocs[ant]
        axyz[1] = Ylocs[ant]
        axyz[2] = elevlocs[ant]
        # Project onto UVW axes.
        antuvws[ant,0] = np.dot(axyz,udir)/np.linalg.norm(udir)
        antuvws[ant,1] = np.dot(axyz,vdir)/np.linalg.norm(vdir)
        antuvws[ant,2] = np.dot(axyz,wdir)/np.linalg.norm(wdir)

#    for ant in range(0,len(Xlocs)):
#        print "%d : %s :  u = %3.4f,  v = %3.4f,  w = %3.4f"%(ant, antnames[ant], antuvws[ant,0]-antuvws[0,0], antuvws[ant,1]-antuvws[0,1], antuvws[ant,2]-antuvws[0,2])

    buvws=[]
    for ant1 in range(0,len(Xlocs)):
        for ant2 in range(ant1+1,len(Xlocs)):
            buvws.append( [ant1, ant2, antuvws[ant2]-antuvws[ant1] ] )
#            if ant1==0:
#                print "%d-%d : %s-%s :  u = %3.4f,  v = %3.4f,  w = %3.4f"%(ant1,ant2,antnames[ant1],antnames[ant2],buvw[0],buvw[1],buvw[2])
        

    ################################
    xaxis = origin + xdir
    yaxis = origin + ydir
    zaxis = origin + zdir
    
    waxis = origin + wdir
    vaxis = origin + vdir
    uaxis = origin + udir
    
    ## Plot XYZ
    #ax.plot( [origin[0], xaxis[0]], [origin[1], xaxis[1]], [origin[2], xaxis[2]], 'c',linewidth=2 )
    #ax.plot( [origin[0], yaxis[0]], [origin[1], yaxis[1]], [origin[2], yaxis[2]], 'c', linewidth=2 )
    ax.plot( [origin[0], zaxis[0]], [origin[1], zaxis[1]], [origin[2], zaxis[2]], 'c' ,linewidth=2)
    #ax.text( xaxis[0], xaxis[1], xaxis[2], '   X' )
    #ax.text( yaxis[0], yaxis[1], yaxis[2], '   Y' )
    ax.text( zaxis[0], zaxis[1], zaxis[2], '   NCP' )

    ## Plot UVW
    #ax.plot( [origin[0], waxis[0]], [origin[1], waxis[1]], [origin[2], waxis[2]], 'r', linewidth=2 )
    #ax.plot( [origin[0], vaxis[0]], [origin[1], vaxis[1]], [origin[2], vaxis[2]], 'r', linewidth=2 )
    #ax.plot( [origin[0], uaxis[0]], [origin[1], uaxis[1]], [origin[2], uaxis[2]], 'r', linewidth=2 )
    #ax.text( waxis[0], waxis[1], waxis[2], 'w' )
    #ax.text( vaxis[0], vaxis[1], vaxis[2], 'v' )
    #ax.text( uaxis[0], uaxis[1], uaxis[2], 'u' )

    ## Draw a star in the source direction
    sf=1.5
    star = sf * (waxis - origin) + origin
    ax.plot( [star[0]], [star[1]], [star[2]], 'y*',markersize=15 )
    
    plt.ion()
    plt.show()
    
    return antuvws, buvws

########################################################

def readAntListFile(antfile=''):
    fp = open(antfile,'r')
    thelines = fp.readlines()
    
    eastlocs=[]
    northlocs=[]
    elevlocs=[]
    antnames=[]
    for aline in thelines:
        if aline[0] != '#':
            words = aline.split()
            antnames.append( words[1] )
            ## make the indices 6,7,8 for listobs files that do not contain pad/station info !!
            ## make the indices 7,8,9 for listobs files that contain antenna pad info !!
            eastlocs.append( eval( words[7] ) )
            northlocs.append( eval( words[8] ) )
            elevlocs.append( eval( words[9] ) )

    #print 'Antenna names : ', antnames
    #print 'East : ', eastlocs
    #print 'North :', northlocs
    #print 'Elev : ',elevlocs
    return np.array(eastlocs), np.array(northlocs), np.array(elevlocs), antnames

###################################################

def ft2d(inpdat):
    idata = np.fft.ifftshift(inpdat)
    fdata=(np.fft.fftn(idata));
    outdat=np.fft.fftshift(fdata);
    return outdat


# Simulate the image seen.
def imlook(hourangle=-3.5, declination=+20.0, obslatitude=34.0,antennalist='', source='point'):

    antuvws, buvws = cdraw(hourangle, declination, obslatitude, antennalist)

    imsize = 256

    truesky = np.zeros( (imsize,imsize), 'float')
    truesky[imsize/2, imsize/2] = 1.0
    truesky[imsize/2+5, imsize/2+5] = 0.8
    truesky[imsize/2+5, imsize/2+10] = 0.5

    truesky[imsize/2+15, imsize/2+30] = 0.6

    truesky[imsize/2+25, imsize/2+45] = 0.6
    truesky[imsize/2+30, imsize/2+50] = 1.0

    truevis = ft2d( truesky)

    measvis = np.zeros( (imsize,imsize), 'complex')

    scaleuv=30.0 ### This is obviously ad-hoc. Proper propagation of units needs to happen here...

    bcnt=0
    for baseline in buvws:
        uvw = baseline[2]
        ucoordp = imsize/2 + uvw[0]/scaleuv
        vcoordp = imsize/2 + uvw[1]/scaleuv
        ucoordm = imsize/2 - uvw[0]/scaleuv
        vcoordm = imsize/2 - uvw[1]/scaleuv

        if ucoordp < 0 or ucoordp > imsize-1 or vcoordp < 0 or vcoordp > imsize-1:
            print 'Baseline outside grid : ', uvw
        else:
            measvis[ int(ucoordp), int(vcoordp) ] = truevis[ int(ucoordp), int(vcoordp) ]
            measvis[ int(ucoordm), int(vcoordm) ] = truevis[ int(ucoordm), int(vcoordm) ]
            bcnt=bcnt+1

    #print 'Used ' , bcnt ,' baselines out of ', len(buvws)

    if bcnt>0.0:
        obsim = ft2d( measvis ) / bcnt

        plt.figure(2)
#        plt.subplot(211)
        plt.imshow( np.real(obsim) )

#        plt.subplot(212)
#        plt.imshow( np.real(truevis) )

    return obsim


def stepants(antennalist='', nants=10):
    fp = open(antennalist)
    antlines = fp.readlines()
    fp.close()

    if len(antlines) < nants:
        nants = len(antlines)

    fname = 'tmplist.txt'

    fp = open(fname,'w')
    for ant in range(0,nants):
        fp.write( antlines[ant] )
    fp.close()

    return fname

def runmovie():

    for nant in range(2,27):
        obsim = imlook(hourangle=-2.5, declination=30.0, antennalist=stepants('vla_ants.txt',nant))
        nsample = 1
        #plt.subplot(212)
        plt.imshow( np.real(obsim) )
        plt.pause(0.1)
    
    for hourangle in np.arange(-3.0,+5.0,1.0):
        obsim = obsim + imlook(hourangle=hourangle, declination=30.0, antennalist='vla_ants.txt')
        nsample = nsample+1
        #plt.subplot(212)
        plt.imshow( np.real(obsim)/nsample )
        
        plt.pause(0.1)


runmovie()