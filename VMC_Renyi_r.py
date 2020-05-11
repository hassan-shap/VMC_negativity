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
#     print(h1,h2)
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
def VMC_main(r,numconfig,V1,wf_r,N,inds_A,N_pt,\
             n_occ_r,n_pos_r,coords_r):

    Ns=4 # number of steps of random walker
    move_attempted = 0
    move_accepted = 0
    
    # possible steps
    step_abs=np.arange(1,Ns+1)
    step_vals=np.sort(np.concatenate((-step_abs,step_abs),axis=0)).tolist()


    inv_wf_r= np.zeros(wf_r.shape,dtype=np.complex128)
    for i_r in range(r):
        inv_wf_r[:,:,i_r]=np.linalg.inv(wf_r[:,:,i_r])

    count_comp=0
    count=0 # counter for energy
    howoften=10 # calculate energy every 10 steps
    min_step=500
    
#     ent_ratio=np.zeros(int(numconfig/howoften),dtype=np.complex64) # total energy
    ent_ratio=np.zeros(numconfig,dtype=np.complex64) # total energy

    
    for step in range(numconfig+min_step):
        for i_r in range(r):
            
            for moved_elec in range(N_pt):
                move_attempted=move_attempted+1

                # random walk of Ns steps left or right
                stepx=random.sample(step_vals,1)[0]
                ptcls_x= np.mod( coords_r[moved_elec,i_r]+stepx, N) # new configuration

                if n_occ_r[ptcls_x,i_r]==1:
                    continue

                pt_wf_1=np.transpose(V1[ptcls_x,:])
                rel=np.dot(inv_wf_r[moved_elec,:,i_r],pt_wf_1)

                alpha=min(1, np.abs(rel)**(2*ex))

                random_num=random.random()

                if random_num <= alpha:
                    u_1=np.reshape(pt_wf_1,(N_pt,1)) - np.reshape(wf_r[:,moved_elec,i_r],(N_pt,1))
                    v=np.zeros((N_pt,1))
                    v[moved_elec]=1
                    inv_wf_r[:,:,i_r]=inv_wf_r[:,:,i_r] - np.dot(np.dot(np.dot(inv_wf_r[:,:,i_r],u_1),v.T),inv_wf_r[:,:,i_r]) \
                                     /(1+np.dot(v.T,np.dot(inv_wf_r[:,:,i_r],u_1))) 
                    wf_r[:,moved_elec,i_r]=np.reshape(pt_wf_1,(N_pt,))

                    move_accepted=move_accepted+1
                    n_occ_r[coords_r[moved_elec,i_r],i_r]= 0
                    n_pos_r[coords_r[moved_elec,i_r],i_r]= 0

                    coords_r[moved_elec,i_r]=ptcls_x
                    n_occ_r[ptcls_x,i_r]= n_occ_r[ptcls_x,i_r]+1
                    n_pos_r[ptcls_x,i_r]= moved_elec+1

            x=np.argwhere(n_pos_r[:,i_r]>0)
            assert len(x)== N_pt, 'no of ptcle is %d' % (len(x))
            assert np.sum(n_occ_r[:,i_r])== N_pt, 'n_occ anc ptcle is %d' % (np.sum(n_occ_r[:,i_r]))

# ##############################################################
            
        if step> (min_step-1):
#             if ((step-min_step+1)%howoften)==0:
            number_pt_inside_A= np.sum(n_occ_r[inds_A,:],axis=0)
            if np.max(number_pt_inside_A)==np.min(number_pt_inside_A) :
                wf_inds= np.zeros((number_pt_inside_A[0],r),dtype=int)

                for i_r in range(r):
                    pt_num_inside = np.argwhere( n_occ_r[inds_A,i_r]>0 )
                    pt_num_inside = np.reshape( pt_num_inside, (len(pt_num_inside),)).tolist()
                    wf_inds[:,i_r]=( n_pos_r[ inds_A[pt_num_inside],i_r ] )-1

                wf_swap = np.zeros(wf_r.shape,dtype=np.complex128)
                for i_r in range(r-1):
                # r permutation of subsystem indices
                    wf_swap[:,:,i_r] = np.copy(wf_r[:,:,i_r])
                    wf_swap[:,wf_inds[:,i_r],i_r] = np.copy(wf_r[:,wf_inds[:,i_r+1],i_r+1])

                wf_swap[:,:,r-1] = np.copy(wf_r[:,:,r-1])
                wf_swap[:,wf_inds[:,r-1],r-1] = np.copy(wf_r[:,wf_inds[:,0],0])

                ratio_r=1.0
                for i_r in range(r):
                    ratio_r *= np.linalg.det(wf_swap[:,:,i_r])/np.linalg.det(wf_r[:,:,i_r])
                ent_ratio[step-min_step] = ratio_r**ex
    #                     ent_ratio[count] = ratio_r
                count_comp+=1
            else:
                ent_ratio[step-min_step] = 0
    #                     ent_ratio[count] = 0
    #                 count+=1

        if (step%500) ==0:
            for i_r in range(r):
                inv_wf_r[:,:,i_r]=np.linalg.inv(wf_r[:,:,i_r])



    acc_ratio=move_accepted/move_attempted
    
    print("Renyi computed %d times" % (count_comp))
    print("Acceptance rate=", acc_ratio)
    return  (np.log(np.mean(ent_ratio)))/(1-r), acc_ratio, count_comp


# initialize wavefunction 
def initialize_wf(N,N_pt):
    coords=np.sort(random.sample(range(N),N_pt))
    n_occ=np.zeros((N,))        
    n_occ[coords]=np.ones((N_pt,))        
    n_pos = np.zeros((N,),dtype=int)        
    n_pos[coords] = np.arange(1,N_pt+1)
    
    return n_occ, n_pos, coords[:N_pt]


ex=2 # 1 is free, 2 is Haldane-Shastry

def main():
    scratch="scratch/"

    r=2 # Renyi index
    # system size
    N=20
    Lsub_list=np.arange(1,N)
#     Lsub_list=np.arange(4,10,2)
    # Lsub_list=[8]
    N_pt = int(N/2)

    # reference slater determinant
    t= -0.
    # hopping amplitudes
    t1= 1-t
    t2= 1+t
    BC=np.exp(1j*pi) # boundary condition on a chain, you can put BC=0 for open chain
    # BC=1 periodic boundary condition and BC=-1 is anti-periodic
    # do not put BC=1 since the gs is not unique in that case
    V1=wf_gen(N,N_pt,BC,t1,t2) # eigenvectors in 

    # print('exact result is ', R2_ex[0])

    Nrep=12
    numconfig=100000

    random.seed(time.time())

    for i_rep in range(8,Nrep):
        print('rep ', i_rep)
        t_timer=time.time()
        
        for i_L in range(len(Lsub_list)):
            Lsub=Lsub_list[i_L]
            print('subsystem size = ', Lsub)
            f1='R%d_e_%d_N_%d_Npt_%d_t_%.2f_Lsub_%d' % (r,ex,N,N_pt,t,Lsub)
            inds_A= np.arange(0,Lsub)

            # ##### initialization
            wf_r= np.zeros((N_pt,N_pt,r),dtype=np.complex64)
            n_occ_r= np.zeros((N,r),dtype=int)
            n_pos_r= np.zeros((N,r),dtype=int)
            coords_r= np.zeros((N_pt,r),dtype=int)

            cond_r= np.zeros(r)
            for i_r in range(r):
                n_occ_r[:,i_r], n_pos_r[:,i_r], coords_r[:,i_r]=initialize_wf(N, N_pt)
                wf_r[:,:,i_r]=np.transpose(V1[coords_r[:,i_r],:])
                cond_r[i_r]=np.linalg.cond(wf_r[:,:,i_r])

            counter=0
            epsilon= 1e-8
            while np.max(cond_r) > 1/epsilon :
                counter+=1
                assert counter<=40, "wave function cannot be constructed!"
                for i_r in range(r):
                    n_occ_r[:,i_r], n_pos_r[:,i_r], coords_r[:,i_r]=initialize_wf(N, N_pt)
                    wf_r[:,:,i_r]=np.transpose(V1[coords_r[:,i_r],:])
                    cond_r[i_r]=np.linalg.cond(wf_r[:,:,i_r])
    
            Rr_vmc, acc_ratio, count_comp=VMC_main(r,numconfig,V1,wf_r,N,inds_A,N_pt,\
                                                   n_occ_r,n_pos_r,coords_r)


            fname= scratch + f1 + '_rep_%d.npz' % (i_rep)
            np.savez(fname, Rr_vmc=Rr_vmc, numconfig=numconfig,\
                     acc_ratio= acc_ratio, count_comp=count_comp )

        elapsed = time.time() - t_timer
        print("VMC finished, elapsed time =", elapsed, "sec")

# Call the main function if the script gets executed (as opposed to imported).
if __name__ == '__main__':
    main()



