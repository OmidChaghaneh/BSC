import math
import numpy as np
from numpy.matlib import repmat
import scipy.signal as ssg    
from scipy.interpolate import interpn
from typing import Tuple
from dataclasses import dataclass

@dataclass
class OutImStruct():
    """Output image structure for scan converted images."""
    scArr: np.ndarray
    xmap: np.ndarray # sc (y,x) --> preSC x
    ymap: np.ndarray # sc (y,x) --> preSC y
    

def iqToRf(iqData, rxFrequency, decimationFactor, carrierFrequency):
    """Convert IQ data to RF data."""
    iqData = ssg.resample_poly(iqData, decimationFactor, 1) # up-sample by decimation factor
    rfData = np.zeros(iqData.shape)
    t = [i*(1/rxFrequency) for i in range(iqData.shape[0])]
    for i in range(iqData.shape[1]):
        rfData[:,i] = np.real(np.multiply(iqData[:,i], np.exp(1j*(2*np.pi*carrierFrequency*np.transpose(t)))))
    return rfData

def scanConvert(inIm: np.ndarray, width: float, tilt: float, startDepth: float, stopDepth: float,
                desiredHeight=500) -> Tuple[OutImStruct, float, float]:
    """ScanConvert sector image
    
    Args:
        inIm (np.ndarray): Input image.
        width (float): Sector width of desired scan conversion image in degrees.
        tilt (float): Tilt of sector image in degrees.
        startDepth (float): Axial depth of first sample in meters.
        stopDepth (float): Axial depth of last sample in meters.
        desiredHeight (int): Desired vertical size of output image in pixels (default 500).
        
    Returns:
        OutImStruct, float, float: Output image, height, and width in centimeters.
    """
    
    #Convert to radians
    samples, beams = inIm.shape
    depthIncrement = (stopDepth-startDepth)/(samples - 1)
    startAngle = np.deg2rad(270 + tilt - width/2)
    angleIncrement = np.deg2rad(width)/(beams-1)

    outIm = inIm
    background = 0
    height = desiredHeight

    # Subtract 180 degrees to get transudcer in top of image if startAngle > pi
    startAngle = startAngle % np.pi
    stopAngle = startAngle + np.deg2rad(width)
    angleRange = np.arange(startAngle, stopAngle+angleIncrement, angleIncrement)

    # Define physical limits of image
    xmin = -1*max(np.cos(startAngle)*np.array([startDepth, stopDepth]))
    xmax = -1*min(np.cos(stopAngle)*np.array([startDepth, stopDepth]))
    ymin = min(np.sin(angleRange)*startDepth)
    ymax = max(np.sin(angleRange)*stopDepth)
    widthScale = abs((xmax-xmin)/(ymax-ymin))
    width = int(np.ceil(height*widthScale))

    heightCm = (ymax-ymin)*100
    widthCm = (xmax-xmin)*100

    # Make (x,y)-plane representation of physical image
    xmat = (np.transpose(np.ones((1, height)))*np.arange(0, width, 1))/(width-1)
    ymat = (np.transpose(np.arange(0, height, 1).reshape((1, height)))*np.ones((1, width)))/(height-1)
    xmat = (xmat*(xmax-xmin)) + xmin
    ymat = (ymat*(ymax-ymin)) + ymin

    # Transform into polar coordinates (angle, range)
    anglemat = np.arctan2(ymat, -1*xmat)
    rmat = np.sqrt((xmat**2) + (ymat**2))

    # Convert phys. angle and range into beam and sample
    anglemat = np.ceil((anglemat - startAngle)/angleIncrement)
    rmat = np.ceil((rmat - startDepth)/depthIncrement)

    # Find pixels outside active sector
    backgr = np.argwhere((rmat<1))
    backgr = np.concatenate((backgr, np.argwhere(rmat>=samples)), axis=0)
    backgr = np.concatenate((backgr, np.argwhere(anglemat<1)), axis=0)
    backgr = np.concatenate((backgr, np.argwhere(anglemat>beams)), axis=0)
    scMap = (anglemat-1)*samples + rmat

    for i in range(backgr.shape[0]):
        scMap[backgr[i,0],backgr[i,1]] = background
    inIm_indx = repmat(np.arange(0, outIm.shape[1]), int(outIm.shape[0]), 1) # <-- maps (y,x) in Iin to indt in Iin
    inIm_indy = np.transpose(repmat(np.arange(0, outIm.shape[0]), int(outIm.shape[1]), 1)) # <-- maps (y,x) in Iout to indr in Iin
        
    outIm = np.append(np.transpose(outIm), background)
    inIm_indy = np.append(np.transpose(inIm_indy), background)
    inIm_indx = np.append(np.transpose(inIm_indx), background)

    scMap = np.array(scMap).astype(np.uint64) - 1
    outIm = outIm[scMap]
    inIm_indy = inIm_indy[scMap]
    inIm_indx = inIm_indx[scMap]

    outIm = np.reshape(outIm, (height, width))
    inIm_indy = np.reshape(inIm_indy, (height, width))
    inIm_indx = np.reshape(inIm_indx, (height, width))
    
    hCm = heightCm
    wCm = widthCm
    
    OutIm = OutImStruct(outIm, inIm_indx, inIm_indy)
    return OutIm, hCm, wCm

def scanConvert3Va(rxLines, lineAngles, planeAngles, beamDist, imgSize, fovSize, z0, interp='linear'):
    pixSizeX = 1/(imgSize[0]-1)
    pixSizeY = 1/(imgSize[1]-1)
    pixSizeZ = 1/(imgSize[2]-1)

    # Create Cartesian grid and convert to polar coordinates
    xLoc = (np.arange(0,1+(pixSizeX/2),pixSizeX)-0.5)*fovSize[0]
    yLoc = (np.arange(0,1+(pixSizeY/2),pixSizeY)-0.5)*fovSize[1]
    zLoc = np.arange(0,1+(pixSizeZ/2),pixSizeZ)*fovSize[2]
    Z, X, Y = np.meshgrid(zLoc, xLoc, yLoc, indexing='ij')

    PHI = np.arctan2(Y, Z+z0)
    TH = np.arctan2(X, np.sqrt(np.square(Y)+np.square(Z+z0)))
    R = np.sqrt(np.square(X)+np.square(Y)+np.square(Z+z0))*(1-z0/np.sqrt(np.square(Y)+np.square(Z+z0)))
    
    zCoords = np.arange(rxLines.shape[0])
    xCoords = np.arange(rxLines.shape[1])   
    yCoords = np.arange(rxLines.shape[2])
    Zcoords, Xcoords, Ycoords = np.meshgrid(zCoords, xCoords, yCoords, indexing='ij')
    coords = Zcoords*rxLines.shape[1]*rxLines.shape[2] + Xcoords*rxLines.shape[2] + Ycoords

    radLineAngles = np.pi*lineAngles/180
    radPlaneAngles = np.pi*planeAngles/180
    
    img = interpn((beamDist, radLineAngles, radPlaneAngles), 
                                    rxLines, (R, TH, PHI), method=interp, bounds_error=False, fill_value=0)
    
    imgCoords = interpn((beamDist, radLineAngles, radPlaneAngles), 
                                    coords, (R, TH, PHI), method='nearest', bounds_error=False, fill_value=-1)
    
    img = np.array(img)
    imgCoords = np.array(imgCoords, dtype=np.int64)

    return img, imgCoords

def scanConvert3dVolumeSeries(dbEnvDatFullVolSeries, scParams, isLin=True, scale=True, interp='linear', normalize=True) -> Tuple[np.ndarray, np.ndarray, list]:
    if len(dbEnvDatFullVolSeries.shape) != 4:
        numVolumes = 1
        nz, nx, ny = dbEnvDatFullVolSeries.shape
    else:
        numVolumes = dbEnvDatFullVolSeries.shape[0]
        nz, nx, ny = dbEnvDatFullVolSeries[0].shape
    apexDist = scParams.VDB_2D_ECHO_APEX_TO_SKINLINE # Distance of virtual apex to probe surface in mm
    azimSteerAngleStart = scParams.VDB_2D_ECHO_START_WIDTH_GC*180/np.pi # Azimuth steering angle (start) in degree
    azimSteerAngleEnd = scParams.VDB_2D_ECHO_STOP_WIDTH_GC*180/np.pi # Azimuth steering angle (end) in degree
    rxAngAz = np.linspace(azimSteerAngleStart, azimSteerAngleEnd, nx) # Steering angles in degree
    elevSteerAngleStart = scParams.VDB_THREED_START_ELEVATION_ACTUAL*180/np.pi # Elevation steering angle (start) in degree
    elevSteerAngleEnd = scParams.VDB_THREED_STOP_ELEVATION_ACTUAL*180/np.pi # Elevation steering angle (end) in degree
    rxAngEl = np.linspace(elevSteerAngleStart, elevSteerAngleEnd, ny) # Steering angles in degree
    DepthMm=scParams.VDB_2D_ECHO_STOP_DEPTH_SIP
    imgDpth = np.linspace(0, DepthMm, nz) # Axial distance in mm
    volDepth = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP *(abs(math.sin(math.radians(elevSteerAngleStart))) + abs(math.sin(math.radians(elevSteerAngleEnd)))) # Elevation (needs validation)
    volWidth = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP *(abs(math.sin(math.radians(azimSteerAngleStart))) + abs(math.sin(math.radians(azimSteerAngleEnd))))   # Lateral (needs validation)
    volHeight = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP - scParams.VDB_2D_ECHO_START_DEPTH_SIP # Axial (needs validation)
    fovSize   = [volWidth, volDepth, volHeight] # [Lateral, Elevation, Axial]
    imgSize = np.array(np.round(np.array([volWidth, volDepth, volHeight])*scParams.pixPerMm), dtype=np.uint32) # [Lateral, Elevation, Axial]

    NonLinThr=3.5e4; NonLinDiv=1.7e4
    LinThr=3e4; LinDiv=3e4
    
    # Generate image
    imgOut = []; imgOutCoords = []
    if numVolumes > 1:
        for k in range(numVolumes):
            rxAngsAzVec = np.linspace(rxAngAz[0],rxAngAz[-1],dbEnvDatFullVolSeries[k].shape[1])
            rxAngsElVec = np.einsum('ikj->ijk', np.linspace(rxAngEl[0],rxAngEl[-1],dbEnvDatFullVolSeries[k].shape[2]))
            curImgOut, curImgOutCoords = scanConvert3Va(dbEnvDatFullVolSeries, rxAngsAzVec, rxAngsElVec, imgDpth,imgSize,fovSize, apexDist, interp=interp)
            if scale:
                if not isLin:
                    imgOut.append((curImgOut-NonLinThr)*255/NonLinDiv)
                else:
                    imgOut.append((curImgOut-LinThr)*255/LinDiv)
            else:
                curImgOut = np.array(curImgOut)
                if normalize:
                    curImgOut /= np.amax(curImgOut)
                    curImgOut *= 255
                imgOut.append(curImgOut)
                imgOutCoords.append(curImgOutCoords)
                
                
        imgOut = np.array(imgOut)
    else:
        rxAngsAzVec = np.linspace(rxAngAz[0],rxAngAz[-1],dbEnvDatFullVolSeries.shape[1])
        rxAngsElVec = np.linspace(rxAngEl[0],rxAngEl[-1],dbEnvDatFullVolSeries.shape[2])
        curImgOut, curImgOutCoords = scanConvert3Va(dbEnvDatFullVolSeries, rxAngsAzVec, rxAngsElVec, imgDpth,imgSize,fovSize, apexDist, interp=interp)
        if scale:
            if not isLin:
                imgOut = (curImgOut-NonLinThr)*255/NonLinDiv
            else:
                imgOut = (curImgOut-LinThr)*255/LinDiv
        else:
            curImgOut = np.array(curImgOut)
            if normalize:
                curImgOut /= np.amax(curImgOut)
                curImgOut *= 255
            imgOut = curImgOut
            imgOutCoords = curImgOutCoords
    
    return imgOut, imgOutCoords, fovSize