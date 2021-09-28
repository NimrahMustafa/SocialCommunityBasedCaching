import numpy as np
import pandas as pd
import minheap as pq
import fptools as fp
import time
import pickle
import math
import omegaIndex
import statistics
import os.path
from os import path
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpmax
from mlxtend.preprocessing import TransactionEncoder

cachingSchemesList = ['Bingo','FIFO','LRU','LFU','MPC','Random'] #DO NOT CHANGE ORDER - or reflect changes in main
cacheCapacityList = [5]#10,15]#,200,400]#[0.005,0.01,0.03,0.05,0.10,0.15,0.20]
testDatasetIDs = [651]#range(636,650+1)
beta_minNumFilesRequested = 3 #community detection parameter
xi_minNumUsersRequestingFile = 4#community identification parameter
gamma_minNumOfCommonCommunities = 2
minCommunitySize = 15
useRealCommunities = True
detectCommUsingDataset = 0
datasetMetadata = pd.read_csv('TestDatasets/Dataset_MetaData.csv', delimiter=',')[
    ['datasetID', 'numFiles','commStructureID']].drop_duplicates(keep='first')

cache = pq.MappedQueue([])  # cache entries are (score,fileID) pairs
fileScoreInCurrCache = {}
numReqServedPerFileInCurrCache = {}
currentTransactions = {}
communityFileLog = {}
evictedFilesNumReqServed = {}
initialScoreOfFile = {}
lastReqIdOfFile = {}

globalVars = {'requestId': 0,
              'cacheCapacity':0,
              'numFiles':0,
              'mistakeIdentifyCount' :0,
              'minCache':[],
              'cachingScores':[],
              'reqCount':0
              }

def readRequestLog(filePath):
    return pd.read_csv(filePath, delimiter=',')[['User','File','Community']]#[:2000]#.drop_duplicates(keep='first')

def transformRequestsToTransactions(requests):
    transactions = list(requests.groupby('File')['User'].apply(list))
    for t in range(len(transactions)):
        transactions[t] = list(set(transactions[t]))
    return transactions

def transformUserInCommunityToCommunitiesOfUser(communities):
    communitiesOfUser = {}
    for i in range(len(communities)):
        for u in communities[i]:
            if u not in communitiesOfUser:
                communitiesOfUser[u] = set()
            communitiesOfUser[u].add(i)
    return communitiesOfUser

def computeFIFOCachingScore(file):
    if file in fileScoreInCurrCache:
        return fileScoreInCurrCache[file]
    else:
        globalVars['requestId']=globalVars['requestId']+1
        return globalVars['requestId']

def computeLRUCachingScore():
    globalVars['requestId'] = globalVars['requestId']+1
    return globalVars['requestId']

def computeLFUCachingScore(file):
    if file in fileScoreInCurrCache:
        return fileScoreInCurrCache[file]+1
    else:
        return 1

def computeMPCCachingScore(file):
    return (file)

def computeRandomCachingScore(file):
    if(file in globalVars['randomCacheList']):
        return 1
    else:
        return 0

def identifyCommunity(user,file,communitiesOfUser,communities):
    identifiedCommunity = 0
    if user not in communitiesOfUser:
        communitiesOfUser[user] = set()
    if (len(currentTransactions[file]) > xi_minNumUsersRequestingFile ):
        #sort users by their degrees
        #so number of intersections performed can be thresholded by the size of resultant set of intersection
        userDegrees = [len(communitiesOfUser[u]) for u in currentTransactions[file]]
        userDegreePairs = list(zip(userDegrees,currentTransactions[file]))
        userDegreePairs.sort(reverse=True)
        usersSortedByDegree = [x[1] for x in userDegreePairs]
        #common set of max size ( even if the users werent sorted as first users's deg is max)
        commonCommunitiesOfRequestingUsers = communitiesOfUser[usersSortedByDegree[0]]
        i=1
        # to ensure intersection does not become empty because of a user or two.
        while(len(commonCommunitiesOfRequestingUsers) > gamma_minNumOfCommonCommunities and i < len(usersSortedByDegree)):
            commonCommunitiesOfRequestingUsers = communitiesOfUser[usersSortedByDegree[i]] & commonCommunitiesOfRequestingUsers
            i+=1
        for c in commonCommunitiesOfRequestingUsers:  # take common community of max size
            if (len(communities[c]) > len(communities[identifiedCommunity])):
                identifiedCommunity = c
    return identifiedCommunity

def computeOurCachingScore(file,identifiedCommunity,communities):
    lastReqIdOfFile[file]=globalVars['reqCount']
    if file not in numReqServedPerFileInCurrCache:
        numReqServedPerFileInCurrCache[file] = 0
    numReqServedPerFileInCurrCache[file] += 1
    cachingScore =1

    if(identifiedCommunity>0):
        initialScoreOfFile[file] = len(communities[identifiedCommunity])
        cachingScore = initialScoreOfFile[file]-numReqServedPerFileInCurrCache[file]
    if file in initialScoreOfFile:
        if (initialScoreOfFile[file] + 10 + xi_minNumUsersRequestingFile > numReqServedPerFileInCurrCache[file]):
            cachingScore = initialScoreOfFile[file] - numReqServedPerFileInCurrCache[file]  # [identifiedCommunity]#-xi_minNumUsersRequestingFile
        else:
            #print('ready to remove file ' + str(file) + ' from record ' + str(
             #   initialScoreOfFile[file]) + ' , ' + str(numReqServedPerFileInCurrCache[file]))
            del currentTransactions[file]
            del initialScoreOfFile[file]
            del numReqServedPerFileInCurrCache[file]
    return cachingScore

def cacheFile(file,cachingScore,cachingSchemeIndex):
    if file in fileScoreInCurrCache:
        cache.update((fileScoreInCurrCache[file], file), (cachingScore, file))
        fileScoreInCurrCache[file] = cachingScore
    else:
        if (len(cache.d) < globalVars['cacheCapacity']):
            cache.push((cachingScore, file))
            fileScoreInCurrCache[file] = cachingScore
            return False
        else:
            if(cachingScore>1):
                globalVars['cachingScore'].append(cachingScore)
            if cachingSchemeIndex==3 or (cachingSchemeIndex!=0 and cache.h[0][0] < cachingScore) or \
                (cachingSchemeIndex==0 and \
                 (cachingScore - cache.h[0][0] < 10 +xi_minNumUsersRequestingFile or \
                  (globalVars['reqCount']-lastReqIdOfFile[cache.h[0][1]] > fileScoreInCurrCache[cache.h[0][1]]))):
                globalVars['minCache'].append(cache.h[0][0])
                removedFile = cache.pop()[1]
                del fileScoreInCurrCache[removedFile]
                cache.push((cachingScore, file))
                fileScoreInCurrCache[file] = cachingScore
            return True

def evaluateCommunityDetectionPerf(communityStructureID,detectedCommunities):
    G = pickle.load(open('CommGen/Pickles/G_Dataset' + str(communityStructureID), 'rb'))
    cSum0 = len([i for i in G.nodes if G.nodes[i]['bipartite']==0])
    groundTruth = {i:list(G[i].keys()) for i in range(cSum0)}
    communities = {k:[i+cSum0-1 for i in detectedCommunities[k]] for k in range(len(detectedCommunities))}
    print(omegaIndex.Omega(communities,groundTruth).omega_score)
    return

def getGroundTruthCommunities(communityStructureID):
    G = pickle.load(open('CommGen/Pickles/G_Dataset' + str(communityStructureID), 'rb'))
    cSum0 = len([i for i in G.nodes if G.nodes[i]['bipartite'] == 0])
    communities  = [[i - cSum0 + 1 for i in G[k].keys()] for k in range(cSum0)]
    return communities

def detectCommunities(requests,filename,cID):
    if not path.exists('RequestLogGraphs/'+filename):
        filesByUsers = requests.groupby('User').filter(lambda x: len(x) > beta_minNumFilesRequested).groupby('User')['File'].apply(set)
        sortedUserList = list(filesByUsers.index)
        numUsers = len(sortedUserList)
        adjac = {}
        for i in range(numUsers):
            adjac[sortedUserList[i]]={}
            for j in range(numUsers):
                if i != j:
                    wght = len(filesByUsers[sortedUserList[i]] & filesByUsers[sortedUserList[j]])
                    if(wght > 0):
                        adjac[sortedUserList[i]][sortedUserList[j]]=wght
        pickle.dump(adjac, open('RequestLogGraphs/'+filename, 'wb'))
    else:
        adjac = pickle.load(open('RequestLogGraphs/'+filename, 'rb'))
    numEdges = sum(len(adjac[u]) for u in adjac.keys())/2
    print(numEdges)
    if not path.exists('DatasetCommunities/' + filename):
        commList = []
        while(numEdges>0):
            maxWeight = 0
            maxWeightEdge = None
            for u in adjac:
                for v in adjac[u]:
                    if(maxWeight < adjac[u][v]):
                        maxWeightEdge = (u,v)
                        maxWeight = adjac[u][v]
            currComm = set()
            currComm.add(maxWeightEdge[0])
            currComm.add(maxWeightEdge[1])
            currCommCutEdgesWght = sum(adjac[maxWeightEdge[0]].values())+sum(adjac[maxWeightEdge[1]].values()) - (2*maxWeight)
            currCommConductance = currCommCutEdgesWght/(currCommCutEdgesWght+maxWeight)
            currCommNeighbourhood = (set(adjac[maxWeightEdge[0]].keys()) | set(adjac[maxWeightEdge[1]].keys())) - currComm
            cont_flag = True
            while(cont_flag and len(currCommNeighbourhood)>0):
                currCommMaxBlngDeg = 0
                currCommMaxBlngDegV = None
                currCommMaxBu = 0

                for u in currCommNeighbourhood:
                    ku=sum(adjac[u].values())
                    bu=0
                    for v in currComm & set(adjac[u].keys()):
                        bu+=adjac[u][v]
                    uBlngDeg = bu/ku
                    if(currCommMaxBlngDeg < uBlngDeg):
                        currCommMaxBlngDeg = uBlngDeg
                        currCommMaxBlngDegV = u
                        currCommMaxBu = bu

                newCommCutEdgesWght = currCommCutEdgesWght + sum(adjac[currCommMaxBlngDegV].values()) - (2*currCommMaxBu)
                newCommConductance = newCommCutEdgesWght/(newCommCutEdgesWght+currCommMaxBu)
                if newCommConductance < currCommConductance: # or check if difference is > 0.001:
                    currComm.add(currCommMaxBlngDegV)
                    currCommNeighbourhood = (currCommNeighbourhood | set(adjac[currCommMaxBlngDegV].keys())) - currComm
                    currCommCutEdgesWght = newCommCutEdgesWght
                    currCommConductance = newCommConductance
                else:
                    cont_flag=False
            for u in currComm:
                for v in currComm & set(adjac[u].keys()):
                    del adjac[u][v]
                    del adjac[v][u]
                    numEdges-=1
            if len(currComm) >= minCommunitySize:
                commList.append(list(currComm))
                print(currComm)
        pickle.dump(commList, open('DatasetCommunities/' + filename, 'wb'))
        plt.hist([len(c) for c in currComm])
        plt.savefig('DatasetCommunities / DetectedCommSizes_' + str(cID) + '.png')
        plt.clf()
    else:
        commList = pickle.load(open('DatasetCommunities/' + filename, 'rb'))
    return commList

def main():
    resultDF = pd.DataFrame(columns=['DatasetID','CacheCap']+cachingSchemesList)
    resultRowId=0
    for testDatasetID in testDatasetIDs:
        print('Dataset ID: '+str(testDatasetID))
        globalVars['numFiles']=datasetMetadata[datasetMetadata['datasetID']==testDatasetID]['numFiles'].values[0]
        requestData = readRequestLog('TestDatasets/RequestLog_Dataset' + str(testDatasetID) + '.csv')
        if useRealCommunities:
            communities=getGroundTruthCommunities(datasetMetadata[datasetMetadata['datasetID']==testDatasetID]['commStructureID'].values[0])
        elif detectCommUsingDataset > 0:
            requestDataToDetectComm = readRequestLog('TestDatasets/RequestLog_Dataset' + str(testDatasetID) + '.csv')
            communities = detectCommunities(requestDataToDetectComm,
                                            'G_' + str(detectCommUsingDataset) + '_b_' + str(beta_minNumFilesRequested),
                                            datasetMetadata[datasetMetadata['datasetID'] == detectCommUsingDataset][
                                                'commStructureID'].values[0])
            #print('# of Communities Detected: ' + str(len(communities)))
            
        
        result=[[0 for k in range(len(cachingSchemesList))] for c in range(len(cacheCapacityList))]
       
        communities.insert(0, [])
        communitiesOfUser = transformUserInCommunityToCommunitiesOfUser(communities)
        requests = list(requestData.itertuples(index=False, name=None))
        for cacheCapIndex in range(len(cacheCapacityList)):
            globalVars['cacheCapacity'] = cacheCapacityList[cacheCapIndex]  # int(globalVars['numFiles']*cacheCapacity)
            globalVars['randomCacheList'] = np.random.randint(low=1, high=globalVars['numFiles'] + 1,
                                                              size=globalVars['cacheCapacity'])
            print('Cache Capacity: '+str(cacheCapacityList[cacheCapIndex]))
            for cachingSchemeIndex in range(len(cachingSchemesList)):
                numCacheHits = 0
                numCacheHitsPer10KReq=0
                print(str('------------------------')+cachingSchemesList[cachingSchemeIndex]+str('------------------------'))
                globalVars['requestId'] = 0
                currentTransactions.clear()
                fileScoreInCurrCache.clear()
                communityFileLog.clear()
                numReqServedPerFileInCurrCache.clear()
                evictedFilesNumReqServed.clear()
                initialScoreOfFile.clear()
                cache.clear()
                reqId=0
                cacheFullFlag=False
                globalVars['minCache']=[]
                globalVars['cachingScore']=[]
                globalVars['reqCount']=0
                lastReqIdOfFile.clear()
                for user, file, comm in requests:
                    globalVars['reqCount']+=1;
                    reqId+=1
                    if file in fileScoreInCurrCache and cacheFullFlag:
                        numCacheHits += 1
                        numCacheHitsPer10KReq+=1

                    if file not in currentTransactions:
                        currentTransactions[file] = []
                    currentTransactions[file].append(user)

                    if cachingSchemeIndex == 0:
                        identifiedCommunity = identifyCommunity(user, file, communitiesOfUser, communities)
                        cachingScore = computeOurCachingScore(file, identifiedCommunity, communities)
                    if cachingSchemeIndex == 1:
                        cachingScore = computeFIFOCachingScore(file)
                    if cachingSchemeIndex == 2:
                        cachingScore = computeLRUCachingScore()
                    if cachingSchemeIndex == 3:
                        cachingScore = computeLFUCachingScore(file)
                    if cachingSchemeIndex == 4:
                        cachingScore = computeMPCCachingScore(file)
                    if cachingSchemeIndex == 5:
                        cachingScore = computeRandomCachingScore(file)

                    if cacheFile(file, cachingScore, cachingSchemeIndex):
                        cacheFullFlag=True
                    if reqId%10000==0 and cachingSchemeIndex==0:
                        print(numCacheHitsPer10KReq*0.01)
                        numCacheHitsPer10KReq=0
                hitRatio = numCacheHits * 100.0 / len(requests)
                result[cacheCapIndex][cachingSchemeIndex]+=hitRatio
        for cacheCapIndex in range(len(cacheCapacityList)):
            resultDF.loc[resultRowId] = [testDatasetID,cacheCapacityList[cacheCapIndex]]+[x/(numberOfWindows-1+1) for x in result[cacheCapIndex]]
            print(resultDF.loc[resultRowId])
            resultRowId+=1
    print(resultDF)
    resultDF.to_csv('Results.csv',mode='a', header=False,index=False)
    return

if __name__== "__main__":
    main()
