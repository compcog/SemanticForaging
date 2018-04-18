# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:50:32 2018

@author: Johnathan Avery
"""

def modelFits(path):
    
    ### LOAD REQUIRED PACKAGES ###
    import numpy as np
    import pandas as pd
    import re

    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=None, names=['SID', 'entry'], delimiter=' ')
    
    #correct behavioral fits
    df = forage.prepareData(df)
    
    ### LOAD SEMANTIC SIMILARITY MATRIX ###
    # (aka 'local cues', here we use cosines from word2vec)
    
    # Similarity labels
    simlab = []
    ofile = open('Data/corpus/similaritylabels.csv','r')#TODO:
    for line in ofile:
        labs = line.split()
        for lab in labs:
            simlab.append(lab)
    ofile.close()
    
    # Similarity values
    simval = np.zeros((len(simlab), len(simlab)))
    ofile = open('Data/corpus/similaritymatrix.csv', 'r')#TODO:
    j=0
    for line in ofile:
        line = re.sub(',\n', '', line)
        sims = line.split(',')
        i=0
        for sim in sims:
            simval[i,j] = sim
            i+=1
        j+=1
    ofile.close()
    
    # Make sure similarity values are non-zero
    for i in range(0,len(simval)):
        for j in range(0,len(simval)):
            if simval[i,j] <= 0:
                simval[i,j] = 0.0001

    ### LOAD FREQUENCY LIST ###
    # (aka 'global cue', using NOW corpus from http://corpus.byu.edu/now/, 4.2 billion words and growing daily)
    
    freqlab = []
    freqval = []
    ofile = open('Data/corpus/frequencies.csv', 'r') #TODO:
    for line in ofile:
        line = re.sub('\n', '', line)
        freqs=line.split(',')
        freqlab.append(freqs[0])
        freqval.append(np.log(float(freqs[1])))
    ofile.close()
    freqval=np.array(freqval)
    
    sidlist = list(set(df['SID']))
    full_entdf = pd.DataFrame()
    full_fitlist = []
    ct = 0
    
    for sid in sidlist:
        ct+=1
        print(str(ct) + '/' + str(len(sidlist)))
    
        # My general initializations
        myfitlist = []
        myentries = np.array(df[df['SID']==sid]['entry'])
        myenttimes = np.array(df[df['SID']==sid].index)
        myused = []
        mytime = []
        
        # For both frequency and similarity metrics:
            # LIST: Metrics corresponding with my observed entries
            # CURRENT: Full metric values, with observed entries becoming 0
            # HISTORY: State of full metric values (ie, "current" during each entry)
        
        # My frequency initializations
        freq_current = np.array(freqval) 
        freq_list = []
        freq_history = []
    
        # My similarity initializations
        sim_current = simval.copy()
        sim_list = []
        sim_history = []
        
        for i in range(0,len(myentries)):
            word = myentries[i]
            if word not in myused:
                
                # Frequency: Get frequency and update relevant lists
                freq_list.append( float(freq_current[freqlab.index(word)]) )
                freq_history.append( np.array(freq_current) )
                freq_current[freqlab.index(word)] = 0.00000001
                
                # Get similarity between this word and preceding word
                if i > 0:
                    sim_list.append( float(sim_current[simlab.index(myentries[i-1]), simlab.index(word)]) )
                    sim_history.append( np.array(sim_current[simlab.index(myentries[i-1]),:]) )                    
                else:
                    sim_list.append(0)
                    sim_history.append( np.array(sim_current[simlab.index(word),:]) )
                sim_current[:,simlab.index(word)] = 0.00000001           
                
                # Update lists
                myused.append(word)
                mytime.append(myenttimes[i])
        
        # Calculate category switches, based on similarity-drop
        myswitch = np.zeros(len(myused))
        for i in range(1,len(myused)-1):
            if (sim_list[i+1] > sim_list[i]) and (sim_list[i-1] > sim_list[i]):
                myswitch[i] = 1    
    
        # Save my entries with corresponding metrics
        mydf = pd.DataFrame({'sid':[sid]*len(myused) , 'ent':myused, 'freq':freq_list, 'sim':sim_list,
                             'switch':myswitch, 'time':mytime},
                            columns=['sid','time','ent','freq','sim','switch'])
        full_entdf = full_entdf.append(mydf)
        
        # Get parameter fits for the different models
        myfitlist.append(sid)
        myfitlist.append(len(myused))
        myfitlist.extend( forage.getfits(freq_list, freq_history, sim_list, sim_history) )
        full_fitlist.append(np.array(myfitlist))
        
    print("Fits Complete.")
    
    # Output data entries with corresponding metrics for visualization in R
    full_entdf = full_entdf.reset_index(drop=True)
    full_entdf.to_csv('Data/results/fullmetrics.csv', index=False, header=True)
    
    # Output parameter & model fits
    full_fitlist = pd.DataFrame(full_fitlist)
    full_fitlist.columns = ['sid', 'nent', 'beta-SF', 'beta-SS', 'merr-SO', 'merr-SR',
                            'beta-DF', 'beta-DS', 'merr-DO', 'merr-DR']
    full_fitlist.to_csv('Data/results/fullfits.csv', index=False, header=True)
    
    
    print("Saved.")


class forage:

    def prepareData(data):
        import pandas as pd
        import re
        # load similarity labels
        simlab = []
        ofile = open('Data/corpus/similaritylabels.csv','r') 
        for line in ofile:
            labs = line.split()
            for lab in labs:
                simlab.append(lab)
        ofile.close()
        
        ### LOAD CORRECTIONS ###
        # This is a look-up list that maps incorrect words onto accepted words that are in the database
        corrections = pd.read_csv('Data/corpus/corrections.txt', header=None, delimiter='\t')
        corrections = corrections.set_index(corrections[0].values)
        corrections.columns = ['_from','_to']
        
        elist = data['entry'].values
        newlist = []
        notfound = []
        
        # Use look-up table to check and correct observed entries
        for ent in elist:
            ent = re.sub(r'\W+', '', ent) # Alphanumericize it
            if ent in simlab:
                # If this entry is appropriate, keep it
                newlist.append(ent)
            elif ent[0:len(ent)-1] in simlab:
                # If this entry is plural, correct to the singular verion
                newlist.append(ent[0:len(ent)-1])
            elif ent in corrections._from:
                # If this entry is correctable, correct it
                newlist.append(corrections.loc[ent]._to)
            else:
                # If this entry is not found in either list, mark for removal and warn user.
                newlist.append('NA')
                notfound.append(ent)
                
        # Remove the rows with inappropriate entries
        data.entry = newlist
        data = data[data.entry!='NA']
        
        # Warn the user of removed entries
        if len(notfound) > 0:
            print('The following items were not found in the database, and were removed: [' +
                  str(len(notfound)) + ' entries removed] \n')
            print(sorted(set(notfound)))
        else:
            print('All items OK.')
        return data[data.entry!='NA']
        # TODO: return statement might not be necessary...
        
    def model_static(beta, freql, freqh, siml, simh):
        import numpy as np
        ct = 0
        for k in range(0, len(freql)):
            if k == 0:
            # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            else:
            # P of item based on frequency and similarity
                numrat = pow(freql[k],beta[0]) * pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0]) * pow(simh[k],beta[1]))
            ct += -np.log(numrat/denrat) # Log likelihood of this item
        return ct
    
    def model_dynamic(beta, freql, freqh, siml, simh):
        import numpy as np
        ct = 0
        for k in range(0, len(freql)):
            if k == 0 :
            # P of item based on frequency alone (freq of this item / freq of all items)
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            elif k > 0 and k < (len(freql)-1) and siml[k+1] > siml[k] and siml[k-1] > siml[k]:
            # If similarity dips, P of item is based again on frequency
                numrat = pow(freql[k],beta[0])
                denrat = sum(pow(freqh[k],beta[0]))
            else:
            # P of item based on combined frequency and similarity
                numrat = pow(freql[k],beta[0])*pow(siml[k],beta[1])
                denrat = sum(pow(freqh[k],beta[0])*pow(simh[k],beta[1]))
            ct += -np.log(numrat/denrat)
        return ct
        
    def getfits( freq_l, freq_h, sim_l, sim_h ):
        import numpy as np
        from scipy.optimize import fmin
    #fmin: Uses a Nelder-Mead simplex algorithm to find the minimum of function of variables.
        r1 = np.random.rand()
        r2 = np.random.rand()
        
    # COMBINED CUE, STATIC MODEL (no dynamic switching, just focusing on two cues with some weights)
        
        # 1.) Optimize model parameters
        v = fmin(forage.model_static, [r1, r2], args=(freq_l, freq_h, sim_l, sim_h), ftol = 0.001, disp=False)
        beta_sf = float(v[0]) # Optimized weight for frequency cue
        beta_ss = float(v[1]) # Optimized weight for similarity cue
        
        # 2.) Determine model fit (errors) at optimal parameters
        mfits = forage.model_static([beta_sf, beta_ss], freq_l, freq_h, sim_l, sim_h)
        
        # 3.) For comparison, determine model fit (errors) without parameter fits
        rfits = forage.model_static([0, 0], freq_l, freq_h, sim_l, sim_h)
        
        
    # COMBINED CUE, DYNAMIC MODEL (switches dynamically between cues)
        
        # 1.) Optimize model parameters
        v = fmin(forage.model_dynamic, [r1,r2], args=(freq_l, freq_h, sim_l, sim_h), ftol = 0.001, disp=False)
        beta_df = float(v[0]) # Optimized weight for frequency cue
        beta_ds = float(v[1]) # Optimized weight for similarity cue
    
        # 2.) Determine model fit (errors) at optimal parameters
        mfitd = forage.model_dynamic([beta_df, beta_ds], freq_l, freq_h, sim_l, sim_h)
        
        # 3.) For comparison, determine model fit (errors) without parameter fits
        rfitd = forage.model_dynamic([0,0], freq_l, freq_h, sim_l, sim_h)
           
        
        results = [ beta_sf, beta_ss, float(mfits), float(rfits),
                   beta_df, beta_ds, float(mfitd), float(rfitd)]
        
        return results
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    