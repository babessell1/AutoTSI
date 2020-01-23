import numpy as np

def urgExtract(ubhFile):
    time = []
    logtime = []
    distance = []
    # organize ubh data
    handle = open(ubhFile, 'r+')
    handle.write("[STOP]\n[STOP]")
    prevLine = ""
    readScan = False
    for line in handle:
        thisLine = line.strip()
        if prevLine == "[timestamp]":
            time.append(int(thisLine))
        if prevLine == "[logtime]" :
            logtime.append(thisLine)
        if prevLine == "[scan]":
            readScan = True
            scanDat = ''
        if readScan and thisLine.startswith('['):
            readScan = False
            addScan = scanDat.split(';')
            addScan = [int(i) for i in addScan]
            distance.append(addScan)
        if readScan:
            scanDat += thisLine
        prevLine = thisLine
    handle.close()
    return time, logtime, distance

def calcTree(distance, stepStart, stepEnd, maxDist, minRatio, maxRatio, noiseLim, minDiam, maxDiam):
    stepsList = []
    distList = []
    dbhList = []
    centerDistList = []
    centerStepList = []

    newSteps = []
    newDist = [] # step, distance
    step = distance[stepStart:stepEnd]
    for j in range(len(step)):
        dist = step[j]
        # noise filter
        if newSteps and np.absolute((dist - newDist[-1])) <= noiseLim:
            newSteps.append(j+stepStart)
            newDist.append(dist)
        elif newSteps and np.absolute((dist - newDist[-1])) > noiseLim:
            if len(newSteps)*newDist[int(len(newDist)/2)] >= minRatio \
            and len(newSteps)*newDist[int(len(newDist)/2)] <= maxRatio \
            and newDist[int(len(newDist)/2)] <= maxDist:
                stepsList.append(newSteps)
                distList.append(newDist)
                newSteps = []
                newDist = []
            else:
                newSteps = []
                newDist = []
        else:
            newSteps = [(j+stepStart)]
            newDist = [dist]
    if stepsList:
        for j in range(len(stepsList)):
            steps = stepsList[j]
            distan = distList[j]
            if 2*abs(distan[0]-distan[-1])/(distan[-1]+distan[0]) < 0.1:
                midStep = int((steps[-1] - steps[0]) / 2) + steps[0]
                C = (distan[-1]+distan[0])*np.sin(np.pi/720*(midStep-steps[0]))
                L = (((distan[-1]+distan[0])/2)**2 - (C/2)**2)**0.5
                midDist = distance[midStep]
                h = L - midDist
                dbh = (4*h**2 + C**2)/(4*h)
                if C/dbh > 0.20 and h/C < 0.5 and h > 0 \
                and abs(midDist-min(distan))/midDist < 0.01:
                    dbhList.append(dbh)
                    centerDistList.append(L)
                    centerStepList.append(midStep)
                else:
                    dbhList.append(0)
                    centerDistList.append(0)
                    centerStepList.append(0)
            else:
                dbhList.append(0)
                centerDistList.append(0)
                centerStepList.append(0)

    deleteIdx = []
    for j in range(len(dbhList)):
        if dbhList[j] < 0 or dbhList[j] < minDiam or dbhList[j] > maxDiam:
            deleteIdx.append(j)
    if deleteIdx:
        stepsList = [i for j, i in enumerate(stepsList) if j not in deleteIdx]
        distList = [i for j, i in enumerate(distList) if j not in deleteIdx]
        dbhList = [i for j, i in enumerate(dbhList) if j not in deleteIdx]
        centerDistList = [i for j, i in enumerate(centerDistList) if j not in deleteIdx]
        centerStepList = [i for j, i in enumerate(centerStepList) if j not in deleteIdx]
    return stepsList, distList, dbhList, centerDistList, centerStepList


def treeTrack(dbhList, centerDistList, centerStepList,
              dbhList1, centerDistList1, centerStepList1,
              dbhList2, centerDistList2, centerStepList2,
              dbhList3, centerDistList3, centerStepList3,
              dir, trackErrLim):

    def closest(lst, val, k): # list, value to compare to, kth closest
        lst = np.asarray(lst)
        idx = (np.abs(lst - val)).argpartition(k)[k]
        return idx

    lst0, lst1, lst2, lst3, isNew = [], [], [], [], []
    passCheck1 = False
    passCheck2 = False
    passCheck3 = False
    failCheck1 = False
    failCheck2 = False
    failCheck3 = False
    for idx0 in range(len(centerStepList)):
        lst0.append(idx0)
        stepVal = centerStepList[idx0]

        k = 0
        while passCheck1 == False and failCheck1 == False:
            if not centerStepList1 or k >= len(centerStepList1):
                lst1.append(np.nan)
                break
            idx1 = closest(centerStepList1, stepVal, k)
            # allow max of 5 steps to pass in timestep, since sensor faces right
            # side, step should always decrease if moving forward
            if ((dir == 'right' and centerStepList1[idx1]-stepVal>=0
            and centerStepList1[idx1]-stepVal<6) \
            or (dir == 'left' and centerStepList1[idx1]-stepVal<=0
            and stepVal-centerStepList1[idx1]<6)) \
            and np.abs(dbhList1[idx1]-dbhList[idx0])/dbhList[idx0] <= trackErrLim \
            and np.abs(centerDistList1[idx1]-centerDistList[idx0])/centerDistList[idx0] <= trackErrLim:
                passCheck1 = True
                lst1.append(idx1)
            elif k <= 3:
                k += 1
            else:
                lst1.append(np.nan)
                failCheck1 = True
                break

        k = 0
        while passCheck2 == False and failCheck2 == False:
            if not centerStepList2 or k >= len(centerStepList2):
                lst2.append(np.nan)
                break
            if passCheck1 == False:
                idx1 = idx0
                stepVal = centerStepList[idx0]
                compDist = centerDistList[idx0]
                compDbh = dbhList[idx0]
            else:
                stepVal = centerStepList1[idx1]
                compDist = centerDistList1[idx1]
                compDbh = dbhList1[idx1]
            idx2 = closest(centerStepList2, stepVal, k)
            # allow max of 5 steps to pass in timestep, since sensor faces right
            # side, step should always decrease if moving forward
            if ((dir == 'right' and centerStepList2[idx2]-stepVal>=0
            and centerStepList2[idx2]-stepVal<6) \
            or (dir == 'left' and centerStepList2[idx2]-stepVal<=0
            and stepVal-centerStepList2[idx2]<6)) \
            and np.abs(dbhList2[idx2]-compDbh)/compDbh <= trackErrLim \
            and np.abs(centerDistList2[idx2]-compDist)/compDist <= trackErrLim:
                passCheck2 = True
                lst2.append(idx2)
            elif k <= 3:
                    k += 1
            else:
                lst2.append(np.nan)
                failCheck2 = True
                break

        k = 0
        while passCheck3 == False and failCheck3 == False:
            if not centerStepList3 or k >= len(centerStepList3):
                lst3.append(np.nan)
                break
            if passCheck2 == False and passCheck1 == False:
                idx2 = idx0
                stepVal = centerStepList[idx0]
                compDist = centerDistList[idx0]
                compDbh = dbhList[idx0]
            elif passCheck2 == False and passCheck1 == True:
                idx2 = idx1
                stepVal = centerStepList1[idx1]
                compDist = centerDistList1[idx1]
                compDbh = dbhList1[idx1]
            else:
                stepVal = centerStepList2[idx2]
                compDist = centerDistList2[idx2]
                compDbh = dbhList2[idx2]
            idx3 = closest(centerStepList3, stepVal, k)
            # allow max of 5 steps to pass in timestep, since sensor faces right
            # side, step should always decrease if moving forward
            if ((dir == 'right' and centerStepList3[idx3]-stepVal>=0
            and centerStepList3[idx3]-stepVal<6) \
            or (dir == 'left' and centerStepList3[idx3]-stepVal<=0
            and stepVal-centerStepList3[idx3]<6)) \
            and np.abs(dbhList3[idx3]-compDbh)/compDbh <= trackErrLim \
            and np.abs(centerDistList3[idx3]-compDist)/compDist <= trackErrLim:
                passCheck3 = True
                lst3.append(idx3)
            elif k <= 3:
                k += 1
            else:
                lst3.append(np.nan)
                failCheck3 = True
                break

        if sum([passCheck1,passCheck2,passCheck3]) ==3 :
            isNew.append(3)
        elif sum([passCheck1,passCheck2,passCheck3]) == 2:
            isNew.append(2)
        elif passCheck1 == True:
            isNew.append(1)
        else:
            isNew.append(0)

    return lst0, lst1, lst2, lst3, isNew
