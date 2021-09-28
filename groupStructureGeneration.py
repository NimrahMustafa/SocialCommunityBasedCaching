import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.community import LFR_benchmark_graph
import time
import pickle
import statistics
import random
from scipy import stats
from collections import Counter
import math
import numbers
import random
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
import pandas as pd
import minheap as pq
import bipartiteGenerator
import scipy.stats as ss
from os import path


def requestLogGeneration():
    communityStructureID=21 # The community structure to use
    newStructure=0 #0 if to be loaded, 1 if new to be generated
    numComm = 100
    commSizeShape = 0.05 #steepness (how much the comm sizes vary) high means fewer larger communities, more smaller communities
    minCommSize = 20
    maxCommSize = 200
    probNewUser = 0.9
    eta = .05
    numCommFileReqPairs = 10000 # int(expectedNumRequests/expectedCommSize)
    batchSize = 25
    numFilesList = [100000]#,5000]#,5000,10000]
    zipFparaList = [0.2,0.4,0.6]#,0.4,0.6]#,0.4,0.6]#,0.6,0.8]#,1.2,1.8]#[0.4,0.8,1.2,1.6,2]
    userRespParamList=[0.1,0.15,0.2,0.25,0.3]#50,100,200,300,400,500]#(500,290),(1000, 540),(2000, 1000)]#,(500,290)]#(100,30),(100,60),(200,60),(200,120),(500,140),(500,290)] #(mean,std) for  mean u and Pr(0<X<2u)=.999, std = u/3.49
    overlapFracWithPrevBatch = 0.5
    userRespTimeDist = 1 # 1 if Normal, 0 if Uniform
    commReqProbDist = 1 # 1 if proportional to size, 0 if uniform
    datasetID=pd.read_csv('TestDatasets/Dataset_MetaData.csv', delimiter=',')\
        ['datasetID'].drop_duplicates(keep='first').max()

    if not newStructure:
        G = pickle.load(open('CommGen/Pickles/G_Dataset' + str(communityStructureID), 'rb'))
        numComm = len([i for i in G.nodes if G.nodes[i]['bipartite'] == 0])
    else:
        commSizeSeqO = [int(i) for i in ((np.random.pareto(a=commSizeShape, size=numComm*100) + 1) * minCommSize) if int(i)<=maxCommSize] #min(maxCommSize, int(i)) for i in
        if(len(commSizeSeqO) < numComm):
            print('Error - len(commSizeSeq) < numComm')
            return

        commSizeSeq = random.sample(commSizeSeqO,k=numComm)
        #sns.distplot(commSizeSeq)
        # plt.hist(commSizeSeq)
        # plt.title('Community Size Distribution Dataset ' + str(communityStructureID))
        # plt.savefig('CommGen/Distributions/CommSizeDist_Dataset' + str(communityStructureID) + '.png')
        # #plt.show()
        # plt.clf()

        G = bipartiteGenerator.preferential_attachment_graph(commSizeSeq, probNewUser, create_using=nx.Graph())
        pickle.dump(G, open('CommGen/Pickles/G_Dataset' + str(communityStructureID), 'wb'))
        '''userDegSizes = []
        for i in range(numComm, len(G.nodes)):
            userDegSizes.append(len(G[i].keys()))
        plt.hist(userDegSizes)
        plt.title('User Membership Distribution for Dataset ' + str(communityStructureID))
        #sns.distplot(userDegSizes)
        plt.title('User Membership Distribution for Dataset ' + str(communityStructureID))
        # plt.show()
        plt.savefig('CommGen/Distributions/UserCommDist_Dataset ' + str(communityStructureID) + '.png')  # '+'N'+str(n)+'P'+str(p)+'.png')
        plt.clf()'''

    datasetIdsList = []
    numUsersList = []
    numRequestsList = []
    numFilesIdxList = []
    zipFparaIdxList = []
    userRespParamIdxList = []

    numUsers = len(G.nodes) - numComm
    commSizes = [len(G[i].keys()) for i in range(numComm)]
    commWeights = [c/sum(commSizes) for c in commSizes]

    for numFilesIdx in range(len(numFilesList)):
        numFiles = numFilesList[numFilesIdx] #500
        for zipFparaIdx in range(len(zipFparaList)):
            zipFpara = zipFparaList[zipFparaIdx]#1.1#0.56
            if zipFpara > 0:
                fileZipF = [i ** (-1 * zipFpara) for i in range(1, numFiles + 1)]
                zeta = sum(fileZipF)
                filePopularity = [fp / zeta for fp in fileZipF]

                if not path.exists('FilePopularity/'+str(numFiles)+'files_'+str(zipFpara)+'zipF'):
                    pickle.dump(filePopularity,open('FilePopularity/'+str(numFiles)+'files_'+str(zipFpara)+'zipF','wb'))
                    # plt.hist(filePopularity, bins=len(filePopularity))
                    # plt.savefig('FilePopularity/' + str(numFiles) + '_' + str(zipFpara) + '.png')
                    # plt.clf()
                else:
                    filePopularity = pickle.load(open('FilePopularity/'+str(numFiles)+'files_'+str(zipFpara)+'zipF', 'rb'))

                if commReqProbDist==1:
                    commFilePairs = set(tuple(zip(random.choices(range(numComm), weights=commWeights, k=numCommFileReqPairs+100),##np.random.randint(low=1,high=numComm,size=numCommFileReqPairs*10),
                                              random.choices(range(numFiles), weights=filePopularity, k=numCommFileReqPairs + 100))))
                else:
                    commFilePairs = set(tuple(zip(np.random.randint(low=1,high=numComm,size=numCommFileReqPairs+100),
                            random.choices(range(numFiles), weights=filePopularity, k=numCommFileReqPairs + 100))))

            else: #uniform file fistribution
                if commReqProbDist==1:
                    commFilePairs = set(tuple(zip(random.choices(range(numComm), weights=commWeights, k=numCommFileReqPairs+100),##np.random.randint(low=1,high=numComm,size=numCommFileReqPairs*10),
                                          np.random.randint(low=1,high=numFiles,size=numCommFileReqPairs+100))))
                else:
                    commFilePairs = set(tuple(zip(np.random.randint(low=1,high=numComm,size=numCommFileReqPairs*10),
                                          np.random.randint(low=1,high=numFiles,size=numCommFileReqPairs+100))))


            if len(commFilePairs) < numCommFileReqPairs:
                print('ERROR - not enough unique community file pairs')
                return

            commFilePairs = list(commFilePairs)[:numCommFileReqPairs]
            for userRespParamIdx in range(len(userRespParamList)):
                datasetID += 1

                if userRespTimeDist:
                    meanUserResp = 0.5#userRespParamList[userRespParamIdx][0]#200 #maxuserResp modelled as a normal distribution
                    varUserResp = 0.15 #userRespParamList[userRespParamIdx][1]#120
                    userResp = np.random.normal(meanUserResp, varUserResp, numUsers)
                    # plt.hist(userResp,bins=100)
                    # plt.title('NumUsers_' + str(numUsers) + '_Resp_N(' + str(meanUserResp) + ',' + str(varUserResp) + ')')
                    # plt.savefig('UserResponseTimes/NumUsers_' + str(numUsers) + '_Resp_N(' + str(meanUserResp) + ',' + str(varUserResp) + ').png')
                    # plt.clf()
                else:
                    scaleUserRespTime = userRespParamList[userRespParamIdx]
                    userResp = np.random.randint(1,scaleUserRespTime,numUsers)

                numRequests = 0
                user = []
                file = []
                time = []
                community = []
                lastTime=0
                countOfUsers = {'early': [], 'normal': [], 'late':[]}
                startTimes={}
                endTimes={}
                for l in range(1,int(numCommFileReqPairs/batchSize)+1):
                    print(l)
                    userBatch = {'dummy': [], 'early': [], 'normal': [], 'late': []}
                    fileBatch = {'early': [], 'normal': [], 'late': []}
                    communityBatch = {'early': [], 'normal': [], 'late': []}
                    userTypes = ['early','normal','late']
                    for i in range(batchSize):#numCommFileReqPairs):
                        pairId=((l-1)*batchSize)+i
                        for x in G[commFilePairs[pairId][0]].keys():
                            uR = userResp[x-numComm]
                            if(uR<=userRespParamList[userRespParamIdx]):
                                userType='early'
                            elif(uR>userRespParamList[userRespParamIdx] and uR<1-userRespParamList[userRespParamIdx]):
                                userType='normal'
                            else:
                                userType='late'
                            userBatch[userType].append(x - numComm + 1)
                            fileBatch[userType].append(numFiles-commFilePairs[pairId][1]) #(commFilePairs[i][1])
                            communityBatch[userType].append(commFilePairs[pairId][0])
                        numRequests += len(G[commFilePairs[pairId][0]].keys())#commSizeSeq[selectedCommunities[i]]

                    startTimes['early'] = lastTime
                    endTimes['early'] = lastTime+len(userBatch['early'])
                    startTimes['normal'] = lastTime+len(userBatch['early'])
                    endTimes['normal'] = lastTime+len(userBatch['early'])+len(userBatch['normal'])
                    startTimes['late'] = lastTime+len(userBatch['early'])+len(userBatch['normal'])
                    endTimes['late'] = lastTime+len(userBatch['early'])+len(userBatch['normal'])+len(userBatch['late'])
                    lastTime = startTimes['normal']+(overlapFracWithPrevBatch*(endTimes['normal']-startTimes['normal']))
                    for j in range(len(userTypes)):
                        user.extend(userBatch[userTypes[j]])
                        file.extend(fileBatch[userTypes[j]])
                        community.extend(communityBatch[userTypes[j]])
                        time.extend(np.random.randint(startTimes[userTypes[j]],endTimes[userTypes[j]],len(userBatch[userTypes[j]])))

                numExtraUsers = int(eta*numUsers)#np.random.randint(minNumExtraUsers,numUsers)
                for i in range(1,numExtraUsers+1):
                    user.append(random.randint(1,numUsers+1))
                    file.append(numFiles-random.choices(range(numFiles), weights=filePopularity, k=1)[0])#(numFiles-np.random.randint(numFiles))#random.sample(fileIds,k=1))#
                    community.append(-1)
                time.extend(np.random.randint(1,lastTime, numExtraUsers))

                df = pd.DataFrame()
                df['Time'] = pd.Series(time)
                df['User'] = pd.Series(user)
                df['Community'] = pd.Series(community)
                df['File'] = pd.Series(file)
                df.sort_values('Time',inplace=True)
                df.reset_index(drop=True,inplace=True)
                datasetIdsList.append(datasetID)
                numUsersList.append(numUsers)
                numRequestsList.append(df.shape[0])
                print('Dataset ID: '+str(datasetID))
                numFilesIdxList.append(numFiles)
                zipFparaIdxList.append(zipFpara)
                userRespParamIdxList.append((meanUserResp,varUserResp,userRespParamList[userRespParamIdx])) #extremeThreshold is the discretizition threshold
                '''if userRespTimeDist:
                    userRespParamIdxList.append((meanUserResp,varUserResp))
                else:
                    userRespParamIdxList.append('shuffle '+str(batchSize))'''
                print('# Users: '+str(numUsers))
                print('# Requests: '+str(df.shape[0]))
                df.to_csv('TestDatasets/RequestLog_Dataset'+str(datasetID)+'.csv', index_label='RequestID')
    dfMetaData=pd.DataFrame()
    dfMetaData['datasetID'] = datasetIdsList
    dfMetaData['numFiles'] = numFilesIdxList
    dfMetaData['zipFpara'] = zipFparaIdxList
    dfMetaData['userResponseParams'] = userRespParamIdxList
    dfMetaData['numUsers'] = numUsersList
    dfMetaData['numRequests'] = numRequestsList
    dfMetaData['commStructureID'] = [communityStructureID] * len(datasetIdsList)
    dfMetaData.to_csv('TestDatasets/Dataset_MetaData.csv', mode='a', header=False,index=False)

def main():
    requestLogGeneration()
    return

if __name__== "__main__":
    main()