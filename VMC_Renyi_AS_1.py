#!/tmp/yes/bin python3

import numpy as np
import random
from math import pi, sqrt
import sys
import os.path
import matplotlib.pyplot as plt
import time


# generates single particle states
def wf_gen(N,N_pt,BC,t1,t2):
    h1=np.ones(N-1)
    h1[0::2]=0
    h2=np.ones(N-1)
    h2[1::2]=0
    hop= np.diag(t1*h1+t2*h2+0j,1)
    hop[N-1,0]= t1*BC
    H_t= -(hop+ np.matrix(hop).H)/2 
    energies, evecs= np.linalg.eigh(H_t)
    return evecs[:,:N_pt]

def exact_renyi_calc(r,GA,epsilon=1e-9):
    chi0, _ =np.linalg.eigh(GA)
    chi1=chi0[np.nonzero(np.abs(chi0)>epsilon)]
    chi2=chi1[np.nonzero(np.abs(chi1-1)>epsilon)]
#     return -np.sum((1-chi2)*np.log(1-chi2)+chi2*np.log(chi2))
    return np.sum(np.log((1-chi2)**r+chi2**r))/(1-r)


# VMC function 
def VMC_equal_number(numconfig,V1,wf_1,wf_1_anc,N,inds_A,N_pt,\
             n_occ1,coords1,\
             n_occ1_anc,coords1_anc):

    move_attempted = 0
    move_accepted = 0

    inv_wf=np.linalg.inv(wf_1)
    inv_wf_anc=np.linalg.inv(wf_1_anc)
        
    count=0 # counter for energy
    howoften=10 # calculate energy every 10 steps
    min_step=500
    ratio_0=1.0+0.0j    
    
    fraction=np.zeros(len(inds_A)+1,dtype=np.float64) 
    
    for step in range(numconfig+min_step):
        for moved_elec in range(N_pt):
            move_attempted=move_attempted+1

            # random walk of one step left or right
            att_num=random.randint(1,2)
            if att_num==1:
                stepx=1
            else:
                stepx=-1

            ptcls_x= np.mod( coords1[moved_elec]+stepx, N) # new configuration

            if n_occ1[ptcls_x]==1:
                continue
            
            pt_wf_1=np.transpose(V1[ptcls_x,:])

            rel=np.dot(inv_wf[moved_elec,:],pt_wf_1)

            alpha=min(1, np.abs(rel)**2)
            
            random_num=random.random()

            if random_num <= alpha:
                u_1=pt_wf_1 - wf_1[:,moved_elec]
                v=np.zeros((N_pt,1))
                v[moved_elec]=1
                inv_wf=inv_wf - np.dot(np.dot(np.dot(inv_wf,u_1),v.T),inv_wf) \
                                 /(1+np.dot(v.T,np.dot(inv_wf,u_1))) 
                wf_1[:,moved_elec]=pt_wf_1
#                 wf_inv=np.linalg.inv(wf_1)
    
             
                move_accepted=move_accepted+1
                n_occ1[coords1[moved_elec]]= n_occ1[coords1[moved_elec]]-1

                coords1[moved_elec]=ptcls_x
                n_occ1[ptcls_x]= n_occ1[ptcls_x]+1
                
        assert np.sum(n_occ1)== N_pt, 'n_occ anc ptcle is %d' % (np.sum(n_occ1))

####################### ancilla ############################

        for moved_elec in range(N_pt):
            move_attempted=move_attempted+1

            # random walk of one step left or right
            att_num=random.randint(1,2)
            if att_num==1:
                stepx=1
            else:
                stepx=-1

            ptcls_x= np.mod( coords1_anc[moved_elec]+stepx, N) # new configuration

            if n_occ1_anc[ptcls_x]==1:
                continue
                            
            pt_wf_1=np.transpose(V1[ptcls_x,:])

            rel=np.dot(inv_wf_anc[moved_elec,:],pt_wf_1)

            alpha=min(1, np.abs(rel)**2)
            
            random_num=random.random()

            if random_num <= alpha:
                u_1=pt_wf_1 - wf_1_anc[:,moved_elec]
                v=np.zeros((N_pt,1))
                v[moved_elec]=1
                inv_wf_anc=inv_wf_anc - np.dot(np.dot(np.dot(inv_wf_anc,u_1),v.T),inv_wf_anc) \
                         /(1+np.dot(v.T,np.dot(inv_wf_anc,u_1))) 
                wf_1_anc[:,moved_elec]=pt_wf_1
#                 wf_inv_anc=np.linalg.inv(wf_1_anc)

                
                move_accepted=move_accepted+1
                n_occ1_anc[coords1_anc[moved_elec]]= n_occ1_anc[coords1_anc[moved_elec]]-1
            
                coords1_anc[moved_elec]=ptcls_x
                n_occ1_anc[ptcls_x]= n_occ1_anc[ptcls_x]+1
            

        assert np.sum(n_occ1_anc)== N_pt, 'n_occ anc ptcle is %d' % (np.sum(n_occ1_anc))


# ##############################################################
            
        if step> (min_step-1):
            Na = np.sum(n_occ1[inds_A])
            Na_anc = np.sum(n_occ1_anc[inds_A])
            if Na==Na_anc:
                fraction[int(Na)] += 1

        if (step%500) ==0:
            inv_wf=np.linalg.inv(wf_1)
            inv_wf_anc=np.linalg.inv(wf_1_anc)


    acc_ratio=move_accepted/move_attempted
    fraction = fraction/numconfig
    
#     print("fractions are ", fraction)
    print("fraction acceptance rate=", acc_ratio)
    return fraction


# VMC function 
def VMC_amplitude_ratio(numconfig,V1,wf_1,wf_1_anc,N,inds_A,N_pt,N_pt_A,\
             n_occ1,n_pos1,coords1,\
             n_occ1_anc,n_pos1_anc,coords1_anc):

    move_attempted = 0
    move_accepted = 0

    inv_wf=np.linalg.inv(wf_1)
    inv_wf_anc=np.linalg.inv(wf_1_anc)

    inside_A = 0
    
    pt_num_inside = np.argwhere( n_occ1[inds_A]>0 )
    pt_num_inside = np.reshape( pt_num_inside, (N_pt_A,)).tolist()
    wf_inds=np.sort( n_pos1[ inds_A[pt_num_inside] ] )-1

    pt_num_inside_anc = np.argwhere( n_occ1_anc[inds_A]>0 )
    pt_num_inside_anc = np.reshape( pt_num_inside_anc, (N_pt_A,)).tolist()
    wf_inds_anc = np.sort( n_pos1_anc[inds_A[pt_num_inside_anc] ] )-1

    # swapping between original wf and ancilla wf
    wf_1_swap = np.copy(wf_1)
    wf_1_swap[:,wf_inds] = wf_1_anc[:,wf_inds_anc]
    wf_1_swap_anc = np.copy(wf_1_anc)  
    wf_1_swap_anc[:,wf_inds_anc] = wf_1[:,wf_inds] 
        
    count=0 # counter for energy
    howoften=10 # calculate SWAP amplitude every 10 steps
    min_step=500
    
    ent_ratio=np.zeros(int(numconfig/howoften),dtype=np.float64) 
    
    for step in range(numconfig+min_step):
        for moved_elec in range(N_pt):
            move_attempted=move_attempted+1

            # random walk of one step left or right
            att_num=random.randint(1,2)
            if att_num==1:
                stepx=1
            else:
                stepx=-1

            ptcls_x= np.mod( coords1[moved_elec]+stepx, N) # new configuration

            if n_occ1[ptcls_x]==1:
                continue

            val_orig = np.min( np.abs(inds_A- coords1[moved_elec]) )
            val_dest = np.min( np.abs(inds_A-ptcls_x) )
            if val_orig==0 and val_dest==0 :
                inside_A=1
                val_1 = np.min(np.abs(wf_inds-moved_elec))
                ind_1 = np.argmin(np.abs(wf_inds-moved_elec))
                assert val_1==0 , 'wrong inds_A for wf'
            elif val_orig!=0 and val_dest!=0 :
                inside_A=0
            else:
                continue
            
            pt_wf_1=np.transpose(V1[ptcls_x,:])

            rel=np.dot(inv_wf[moved_elec,:],pt_wf_1)

            alpha=min(1, np.abs(rel)**2)
            
            random_num=random.random()

            if random_num <= alpha:
                u_1=np.reshape(pt_wf_1,(N_pt,1))  - np.reshape(wf_1[:,moved_elec],(N_pt,1))
                v=np.zeros((N_pt,1))
                v[moved_elec]=1
                inv_wf=inv_wf - np.dot(np.dot(np.dot(inv_wf,u_1),v.T),inv_wf) \
                                 /(1+np.dot(v.T,np.dot(inv_wf,u_1))) 
                wf_1[:,moved_elec]=np.reshape(pt_wf_1,(N_pt,))
#                 wf_inv=np.linalg.inv(wf_1)
    
                if inside_A==1:
#                     print(wf_1_swap_anc[:,wf_inds_anc[ind_1]].shape)
#                     print(pt_wf_1.shape)
                    wf_1_swap_anc[:,wf_inds_anc[ind_1]]= np.reshape(pt_wf_1,(N_pt,))
                    inside_A=0
                else:
#                     print(wf_1_swap[:,moved_elec].shape)
#                     print(pt_wf_1.shape)
                    wf_1_swap[:,moved_elec]= np.reshape(pt_wf_1,(N_pt,))
                
                move_accepted=move_accepted+1
                n_occ1[coords1[moved_elec]]= n_occ1[coords1[moved_elec]]-1
                n_pos1[coords1[moved_elec]]= 0

                coords1[moved_elec]=ptcls_x
                n_occ1[ptcls_x]= n_occ1[ptcls_x]+1
                n_pos1[ptcls_x]= moved_elec+1
                
        x=np.argwhere(n_pos1>0)
        assert len(x)== N_pt, 'no of ptcle is %d' % (len(x))
        assert np.sum(n_occ1[inds_A])== N_pt_A, 'n_occ_A anc ptcle is %d' % (np.sum(n_occ1[inds_A]))
        assert np.sum(n_occ1)== N_pt, 'n_occ anc ptcle is %d' % (np.sum(n_occ1))

####################### ancilla ############################

        for moved_elec in range(N_pt):
            move_attempted=move_attempted+1

            # random walk of one step left or right
            att_num=random.randint(1,2)
            if att_num==1:
                stepx=1
            else:
                stepx=-1

            ptcls_x= np.mod( coords1_anc[moved_elec]+stepx, N) # new configuration

            if n_occ1_anc[ptcls_x]==1:
                continue
            
            val_orig = np.min( np.abs(inds_A- coords1_anc[moved_elec]) )
            val_dest = np.min( np.abs(inds_A-ptcls_x) )
            if val_orig==0 and val_dest==0 :
                inside_A = 1
                val_1 = np.min(np.abs(wf_inds_anc-moved_elec))
                ind_1 = np.argmin(np.abs(wf_inds_anc-moved_elec))
#                 print(ind_1,val_1)
                assert val_1==0 , 'wrong inds_A for anc wf'
            elif val_orig!=0 and val_dest!=0 :
                inside_A=0
            else:
                continue
                
            pt_wf_1=np.transpose(V1[ptcls_x,:])

            rel=np.dot(inv_wf_anc[moved_elec,:],pt_wf_1)

            alpha=min(1, np.abs(rel)**2)
            
            random_num=random.random()

            if random_num <= alpha:
                u_1=np.reshape(pt_wf_1,(N_pt,1))  - np.reshape(wf_1_anc[:,moved_elec],(N_pt,1))
                v=np.zeros((N_pt,1))
                v[moved_elec]=1
                inv_wf_anc=inv_wf_anc - np.dot(np.dot(np.dot(inv_wf_anc,u_1),v.T),inv_wf_anc) \
                         /(1+np.dot(v.T,np.dot(inv_wf_anc,u_1))) 
                wf_1_anc[:,moved_elec]=np.reshape(pt_wf_1,(N_pt,))
#                 wf_inv_anc=np.linalg.inv(wf_1_anc)

                if inside_A==1:
                    wf_1_swap[:,wf_inds[ind_1]]= np.reshape(pt_wf_1,(N_pt,))
                    inside_A=0
                else:
                    wf_1_swap_anc[:,moved_elec]= np.reshape(pt_wf_1,(N_pt,))
                
                move_accepted=move_accepted+1
                n_occ1_anc[coords1_anc[moved_elec]]= n_occ1_anc[coords1_anc[moved_elec]]-1
                n_pos1_anc[coords1_anc[moved_elec]]= 0

                coords1_anc[moved_elec]=ptcls_x
                n_occ1_anc[ptcls_x]= n_occ1_anc[ptcls_x]+1
                n_pos1_anc[ptcls_x]= moved_elec+1


        x=np.argwhere(n_pos1_anc>0)
        assert len(x)== N_pt, 'no of anc ptcle is %d' % (len(x))
        assert np.sum(n_occ1_anc[inds_A])== N_pt_A, 'n_occ_A anc ptcle is %d' % (np.sum(n_occ1_anc[inds_A]))
        assert np.sum(n_occ1_anc)== N_pt, 'n_occ anc ptcle is %d' % (np.sum(n_occ1_anc))
#         print(n_pos1)

# ##############################################################
        assert n_occ1.sum()==N_pt, "total particle number changes!!"      
            
        if step> (min_step-1):

            if ((step-min_step+1)%howoften)==0:
                ent_ratio[count] = np.abs(np.linalg.det(wf_1_swap)*np.linalg.det(wf_1_swap_anc)/\
                                        (np.linalg.det(wf_1)*np.linalg.det(wf_1_anc)))
                count+=1
        
        if (step%500) ==0:
            inv_wf=np.linalg.inv(wf_1)
            inv_wf_anc=np.linalg.inv(wf_1_anc)


    acc_ratio=move_accepted/move_attempted
    
    
    print("Amplitude acceptance rate=", acc_ratio)
    return np.mean(ent_ratio)


# VMC function 
def VMC_phase_ratio(numconfig,V1,wf_1,wf_1_anc,N,inds_A,N_pt,N_pt_A,\
             n_occ1,n_pos1,coords1,\
             n_occ1_anc,n_pos1_anc,coords1_anc):

    move_attempted = 0
    move_accepted = 0

    inv_wf=np.linalg.inv(wf_1)
    inv_wf_anc=np.linalg.inv(wf_1_anc)

    inside_A = 0
    
    pt_num_inside = np.argwhere( n_occ1[inds_A]>0 )
    pt_num_inside = np.reshape( pt_num_inside, (N_pt_A,)).tolist()
    wf_inds=np.sort( n_pos1[ inds_A[pt_num_inside] ] )-1

    pt_num_inside_anc = np.argwhere( n_occ1_anc[inds_A]>0 )
    pt_num_inside_anc = np.reshape( pt_num_inside_anc, (N_pt_A,)).tolist()
    wf_inds_anc = np.sort( n_pos1_anc[inds_A[pt_num_inside_anc] ] )-1

    # swapping between original wf and ancilla wf
    wf_1_swap = np.copy(wf_1)
    wf_1_swap[:,wf_inds] = wf_1_anc[:,wf_inds_anc]
    wf_1_swap_anc = np.copy(wf_1_anc)  
    wf_1_swap_anc[:,wf_inds_anc] = wf_1[:,wf_inds] 
    
    inv_wf_1_swap = np.linalg.inv(wf_1_swap)
    inv_wf_1_swap_anc = np.linalg.inv(wf_1_swap_anc)

    
    min_step=500
    
    phi_0=np.angle( np.conj(np.linalg.det(wf_1)*np.linalg.det(wf_1_anc))*\
                   np.linalg.det(wf_1_swap)*np.linalg.det(wf_1_swap_anc) )    
    phi=phi_0
#     phi=np.exp(1j*phi_0)

    ent_ratio=np.zeros(numconfig,dtype=np.complex64) 
    
    for step in range(numconfig+min_step):
        for moved_elec in range(N_pt):
            move_attempted=move_attempted+1

            # random walk of one step left or right
            att_num=random.randint(1,2)
            if att_num==1:
                stepx=1
            else:
                stepx=-1

            ptcls_x= np.mod( coords1[moved_elec]+stepx, N) # new configuration

            if n_occ1[ptcls_x]==1:
                continue

            pt_wf_1=np.transpose(V1[ptcls_x,:])

            val_orig = np.min( np.abs(inds_A- coords1[moved_elec]) )
            val_dest = np.min( np.abs(inds_A-ptcls_x) )
            if val_orig==0 and val_dest==0 :
                inside_A=1
                val_1 = np.min(np.abs(wf_inds-moved_elec))
                ind_1 = np.argmin(np.abs(wf_inds-moved_elec))
                assert val_1==0 , 'wrong inds_A for wf'
                rel = np.dot(inv_wf_1_swap_anc[wf_inds_anc[ind_1],:],pt_wf_1)
            elif val_orig!=0 and val_dest!=0 :
                inside_A=0
                rel = np.dot(inv_wf_1_swap[moved_elec,:],pt_wf_1)
            else:
                continue
            

            rel=rel*np.conj(np.dot(inv_wf[moved_elec,:],pt_wf_1))

            alpha=min(1, np.abs(rel)**2)
            
            random_num=random.random()

            if random_num <= alpha:
                u_1=np.reshape(pt_wf_1,(N_pt,1))  - np.reshape(wf_1[:,moved_elec],(N_pt,1))
                v=np.zeros((N_pt,1))
                v[moved_elec]=1
                inv_wf=inv_wf - np.dot(np.dot(np.dot(inv_wf,u_1),v.T),inv_wf) \
                                 /(1+np.dot(v.T,np.dot(inv_wf,u_1))) 
                wf_1[:,moved_elec]=np.reshape(pt_wf_1,(N_pt,))
#                 wf_inv=np.linalg.inv(wf_1)

                if inside_A==1:
                    u_1=np.reshape(pt_wf_1,(N_pt,1)) - np.reshape(wf_1_swap_anc[:,wf_inds_anc[ind_1]],(N_pt,1))
                    v=np.zeros((N_pt,1))
                    v[wf_inds_anc[ind_1]]=1
                    inv_wf_1_swap_anc=inv_wf_1_swap_anc - np.dot(np.dot(np.dot(inv_wf_1_swap_anc,u_1),v.T),inv_wf_1_swap_anc) \
                                 /(1+np.dot(v.T,np.dot(inv_wf_1_swap_anc,u_1))) 
                    wf_1_swap_anc[:,wf_inds_anc[ind_1]]= np.reshape(pt_wf_1,(N_pt,))
                    inside_A=0
                else:
                    u_1=np.reshape(pt_wf_1,(N_pt,1)) - np.reshape(wf_1_swap[:,moved_elec],(N_pt,1))
                    inv_wf_1_swap=inv_wf_1_swap - np.dot(np.dot(np.dot(inv_wf_1_swap,u_1),v.T),inv_wf_1_swap) \
                                 /(1+np.dot(v.T,np.dot(inv_wf_1_swap,u_1))) 

                    wf_1_swap[:,moved_elec]= np.reshape(pt_wf_1,(N_pt,))
                
#                 phi=phi*np.exp(1j*np.angle(rel))
                phi=phi+np.angle(rel)
                
                move_accepted=move_accepted+1
                n_occ1[coords1[moved_elec]]= n_occ1[coords1[moved_elec]]-1
                n_pos1[coords1[moved_elec]]= 0

                coords1[moved_elec]=ptcls_x
                n_occ1[ptcls_x]= n_occ1[ptcls_x]+1
                n_pos1[ptcls_x]= moved_elec+1
                
        x=np.argwhere(n_pos1>0)
        assert len(x)== N_pt, 'no of ptcle is %d' % (len(x))
        assert np.sum(n_occ1)== N_pt, 'n_occ anc ptcle is %d' % (np.sum(n_occ1))

####################### ancilla ############################

        for moved_elec in range(N_pt):
            move_attempted=move_attempted+1

            # random walk of one step left or right
            att_num=random.randint(1,2)
            if att_num==1:
                stepx=1
            else:
                stepx=-1

            ptcls_x= np.mod( coords1_anc[moved_elec]+stepx, N) # new configuration

            if n_occ1_anc[ptcls_x]==1:
                continue
            
            pt_wf_1=np.transpose(V1[ptcls_x,:])

            val_orig = np.min( np.abs(inds_A- coords1_anc[moved_elec]) )
            val_dest = np.min( np.abs(inds_A-ptcls_x) )
            if val_orig==0 and val_dest==0 :
                inside_A = 1
                val_1 = np.min(np.abs(wf_inds_anc-moved_elec))
                ind_1 = np.argmin(np.abs(wf_inds_anc-moved_elec))
#                 print(ind_1,val_1)
                assert val_1==0 , 'wrong inds_A for anc wf'
                rel = np.dot(inv_wf_1_swap[wf_inds[ind_1],:],pt_wf_1)
            elif val_orig!=0 and val_dest!=0 :
                inside_A=0
                rel = np.dot(inv_wf_1_swap_anc[moved_elec,:],pt_wf_1)
            else:
                continue
                

            rel=rel*np.conj(np.dot(inv_wf_anc[moved_elec,:],pt_wf_1))

            alpha=min(1, np.abs(rel)**2)
            
            random_num=random.random()

            if random_num <= alpha:
                u_1=np.reshape(pt_wf_1,(N_pt,1))  - np.reshape(wf_1_anc[:,moved_elec],(N_pt,1))
                v=np.zeros((N_pt,1))
                v[moved_elec]=1
                inv_wf_anc=inv_wf_anc - np.dot(np.dot(np.dot(inv_wf_anc,u_1),v.T),inv_wf_anc) \
                                 /(1+np.dot(v.T,np.dot(inv_wf_anc,u_1))) 
                wf_1_anc[:,moved_elec]=np.reshape(pt_wf_1,(N_pt,))
#                 wf_inv_anc=np.linalg.inv(wf_1_anc)

                if inside_A==1:
                    u_1=np.reshape(pt_wf_1,(N_pt,1)) - np.reshape(wf_1_swap[:,wf_inds_anc[ind_1]],(N_pt,1))
                    v=np.zeros((N_pt,1))
                    v[wf_inds[ind_1]]=1
                    inv_wf_1_swap=inv_wf_1_swap - np.dot(np.dot(np.dot(inv_wf_1_swap,u_1),v.T),inv_wf_1_swap) \
                                /(1+np.dot(v.T,np.dot(inv_wf_1_swap,u_1))) 

                    wf_1_swap[:,wf_inds[ind_1]]= np.reshape(pt_wf_1,(N_pt,))
                    inside_A=0

                else:
                    u_1=np.reshape(pt_wf_1,(N_pt,1)) - np.reshape(wf_1_swap_anc[:,moved_elec],(N_pt,1))
                    inv_wf_1_swap_anc=inv_wf_1_swap_anc - np.dot(np.dot(np.dot(inv_wf_1_swap_anc,u_1),v.T),inv_wf_1_swap_anc) \
                                 /(1+np.dot(v.T,np.dot(inv_wf_1_swap_anc,u_1))) 

                    wf_1_swap_anc[:,moved_elec]= np.reshape(pt_wf_1,(N_pt,))
                
#                 phi=phi*np.exp(1j*np.angle(rel))
                phi=phi+np.angle(rel)


                move_accepted=move_accepted+1
                n_occ1_anc[coords1_anc[moved_elec]]= n_occ1_anc[coords1_anc[moved_elec]]-1
                n_pos1_anc[coords1_anc[moved_elec]]= 0

                coords1_anc[moved_elec]=ptcls_x
                n_occ1_anc[ptcls_x]= n_occ1_anc[ptcls_x]+1
                n_pos1_anc[ptcls_x]= moved_elec+1


        x=np.argwhere(n_pos1_anc>0)
        assert len(x)== N_pt, 'no of anc ptcle is %d' % (len(x))
        assert np.sum(n_occ1_anc)== N_pt, 'n_occ anc ptcle is %d' % (np.sum(n_occ1_anc))


# ##############################################################
        assert n_occ1.sum()==N_pt, "total particle number changes!!"      
            
        if step> (min_step-1):
            ent_ratio[step-min_step] = phi
            
        if (step%500) ==0:
            inv_wf=np.linalg.inv(wf_1)
            inv_wf_anc=np.linalg.inv(wf_1_anc)
            inv_wf_1_swap=np.linalg.inv(wf_1_swap)
            inv_wf_1_swap_anc=np.linalg.inv(wf_1_swap_anc)

    acc_ratio=move_accepted/move_attempted
        
    print("Angle acceptance rate=", acc_ratio)
    return np.mean(np.exp(1j*ent_ratio))
#     return np.mean(ent_ratio)


def Renyi_vmc_runner(numconfig,V1,N, N_pt,Lsub):
    
    inds_A= np.arange(0,Lsub)

    ent_amp=np.zeros(Lsub+1)
    ent_phase=np.zeros(Lsub+1,dtype=np.complex64)

    Na_list=np.arange(int(Lsub/2)-1,int(Lsub/2)+2)

    for N_pt_A in Na_list:    
        print(N_pt_A)
        n_occ0, n_pos0, coords0=initialize_wf(N, N_pt, inds_A, N_pt_A)
        wf_0=np.transpose(V1[coords0,:])
        n_occ0_anc,n_pos0_anc, coords0_anc=initialize_wf(N, N_pt, inds_A, N_pt_A)
        wf_0_anc=V1[coords0_anc,:]

        epsilon= 1e-8
        counter=0
        while np.linalg.cond(wf_0) > 1/epsilon or np.linalg.cond(wf_0_anc) > 1/epsilon:
            counter+=1
            assert counter<=40, "wave function cannot be constructed!"
            n_occ0, n_pos0, coords0=initialize_wf(N, N_pt, inds_A, N_pt_A)
            wf_0=np.transpose(V1[coords0,:])
            n_occ0_anc,n_pos0_anc, coords0_anc=initialize_wf(N, N_pt, inds_A, N_pt_A)
            wf_0_anc=V1[coords0_anc,:]

        ent_phase[N_pt_A]=VMC_phase_ratio(numconfig,V1,np.copy(wf_0),np.copy(wf_0_anc),\
                                          N,inds_A,N_pt,N_pt_A,\
                                          np.copy(n_occ0),np.copy(n_pos0),np.copy(coords0),\
                                          np.copy(n_occ0_anc),np.copy(n_pos0_anc),np.copy(coords0_anc))

        ent_amp[N_pt_A]=VMC_amplitude_ratio(numconfig,V1,np.copy(wf_0),np.copy(wf_0_anc),\
                                          N,inds_A,N_pt,N_pt_A,\
                                          np.copy(n_occ0),np.copy(n_pos0),np.copy(coords0),\
                                          np.copy(n_occ0_anc),np.copy(n_pos0_anc),np.copy(coords0_anc))


    n_occ0, n_pos0, coords0=initialize_wf(N, N_pt, inds_A, N_pt_A)
    wf_0=np.transpose(V1[coords0,:])
    n_occ0_anc,n_pos0_anc, coords0_anc=initialize_wf(N, N_pt, inds_A, N_pt_A)
    wf_0_anc=V1[coords0_anc,:]
    ent_fr=VMC_equal_number(numconfig,V1,wf_0,wf_0_anc,N,inds_A,N_pt,\
             n_occ0,coords0,\
             n_occ0_anc,coords0_anc)

#     ent_vmc= -np.log(np.sum(ent_fr*ent_amp*ent_phase))

    return ent_fr, ent_amp, ent_phase

# initialize wavefunction 
def initialize_wf(N, N_pt, inds_A, N_pt_A):
    
    inds_A = inds_A.tolist()
    inds_outsideA = np.arange(N)
    inds_outsideA = np.delete(inds_outsideA, inds_A).tolist()
    N_pt_remainder = N_pt-N_pt_A

    assert N_pt_A <= len(inds_A), 'N_pt_A > subsystem A size'
    assert N_pt_remainder <= N-len(inds_A), 'N_pt_B > subsystem B size'

    coords = [0]*N_pt       
    coords[:N_pt_A] = np.sort(random.sample(inds_A, N_pt_A))
    coords[N_pt_A:] = np.sort(random.sample(inds_outsideA, N_pt_remainder))
#     print(coords)
    n_occ = np.zeros((N,))        
    n_occ[coords] = np.ones((N_pt,))    
    n_pos = np.zeros((N,),dtype=int)        
    n_pos[coords] = np.arange(1,N_pt+1)
    
    return n_occ, n_pos, coords[:N_pt]


def main():
    scratch="output_files/"

    # system size
    N=20
    # Lsub_list=np.arange(2,N-1)
    # Lsub_list=[7]
    Lsub_list=np.arange(2,10,2)
    N_pt = int(N/2)

    # reference slater determinant
    t= 0.5 
    # hopping amplitudes
    t1= 1-t
    t2= 1+t
    BC=np.exp(1j*pi) # boundary condition on a chain, you can put BC=0 for open chain
    # BC=1 periodic boundary condition and BC=-1 is anti-periodic
    # do not put BC=1 since the gs is not unique in that case
    V1=wf_gen(N,N_pt,BC,t1,t2) # eigenvectors in 

    numconfig=10000
    t_timer=time.time()

    random.seed(time.time())
    # #################################################################
    # ##### Execution Part
    # #################################################################
    R2_ex=np.zeros(len(Lsub_list))
    R2_vmc=np.zeros(len(Lsub_list), dtype=np.complex64)
    Gmat=np.dot(V1,np.matrix(V1).H)

    for i_L in range(len(Lsub_list)):
        Lsub=Lsub_list[i_L]
        print('subsystem size = ', Lsub)
        inds_A= np.arange(0,Lsub_list[i_L])

        ent_fr, ent_amp, ent_phase=Renyi_vmc_runner(numconfig,V1,N, N_pt,Lsub)
        
        np.savez(fname, ent_fr=ent_fr, ent_amp=ent_amp, ent_phase=ent_phase)


    print('VMC Renyi is', R2_vmc)
    print('exact result is ', R2_ex)
    

    elapsed = time.time() - t_timer
    print("VMC finished, elapsed time =", elapsed, "sec")


# Call the main function if the script gets executed (as opposed to imported).
if __name__ == '__main__':
    main()

