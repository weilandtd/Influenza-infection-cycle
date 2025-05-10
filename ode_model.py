import numpy as np 
from collections import OrderedDict

R = 8.314e-3 # kJ/(mol*K)
T0 = 273.15 + 37 # K
NA = 6.022e23 # 1/mol
V = 1e-15 #5 # L Nuclear volume 10^-13 TODO Check this value   

# Protein binding -> uM
# RNA binding -> nM 
# NP Binding to Sc < than NP binding to Sv

# Assume constant pool of cappedRNA that is degraded during the infection

# 

# This is a model for the virus replication and transciption 
# Basic molecules are VRNA, CRNA, MRNA, Pol, NP, MCRNA, MVRNA
# Which form the following complexes:

# Segment complexes with Pol and Np for replication
# Sv + Pol <=dG_Pol=> Sv_Pol + NP -dG_Pol-> Sv_Pol_NP -R-> Sv + Np + Sc
# Sc + Pol <=dG_Pol=> Sc_Pol + NP -dG_Pol-> Sc_Pol_NP -R-> Sc + NP + Sv

# Transcription of Sv into mRNA
# Sv needs a cap from capped host RNA to make mRNA
# Sv + Cap -T-> mRNA + Sv 

# Translation n maybe 20
# mRNA -TL-> mRNA + Pol + n * NP

# Degradation of mRNA
# mRNA -DEG-> 0

# Replication of miniviral RNAs
# Sv_Pol -R-> MCRNA + Pol + Sv
# Sc_Pol -R-> MVRNA + Pol + Sc
# MCRNA + Pol <=dG_Mini=> MCRNA_Pol -R-> MCRNA + Pol + MCRNA
# MVRNA + Pol <=dG_Mini=> MVRNA_Pol -R-> MVRNA + Pol + MVRNA

# RNA Degradation ?! 
# mRNA -DEG-> 0
# MCRNA -DEG-> 0
# MVRNA -DEG-> 0

# Protein Degradation ?!
# Pol -DEG-> 0
# NP -DEG-> 0

# Parameters
# R -> Replication rate constant (1/min)
# T -> Transcription rate constant (1/min)
# TL -> Translation rate constant (1/min)
# DEF -> degradation rate constant (1/min)
# dG_XXX -> Binding energy of NP or RNA molecule to Polymerase 

# The binding of complexes is diffusion limited with the bimolecular rate constant k (1/M/min)

DG_NAMES = ['dG_vRNA', 'dG_cRNA', 'dG_pol', 'dG_np' ,'dG_cap', ] #'dG_DS']

def model(y, t, p):

    # Compute binding constants
    deltaG = np.array([p[g] for g in DG_NAMES])
    T = p['T']
    K = binding_constants( deltaG, T)
    K = {g: K[i] for i, g in enumerate(DG_NAMES)}

    # Compute reaction stoichiometry
    S = build_reaction_stoichiometry(p).T

    # Unpack state variables
    cap = y[0]
    Sv = y[1]
    Sc = y[2]
    Sv_Pol = y[3]
    Sc_Pol = y[4]
    Sv_Pol_Np = y[5]
    Sc_Pol_Np = y[6]
    MCRNA = y[7]
    MVRNA = y[8]
    MCRNA_Pol_Pol = y[9]
    MVRNA_Pol_Pol = y[10]
    mRNA_Pol = y[11]
    mRNA_NP = y[12]
    Pol = y[13]
    NP = y[14]
    Sv_c = y[15]
    MVRNA_Pol= y[16]
    MCRNA_Pol = y[17]
    cappedMVRNA = y[18] 
    cappedMCRNA = y[19]
    dsMVRNA = y[20]
    mRNA_NEP = y[21]
    NEP = y[22]

    # Compute replication rates
    R_Sc = p['krc'] * Sv_Pol_Np * NP/(NP + K['dG_np'])
    R_Sv = p['krv'] * Sc_Pol_Np * NP/(NP + K['dG_np'])

    # Initiation of mini replication
    R_I_MCRNA = p['krm'] * Sv_Pol

    # Mini self replication
    R_MCRNA = p['krc'] * MVRNA_Pol_Pol
    R_MVRNA = p['krv'] * MCRNA_Pol_Pol
    
    # Compute binding rates
    B_Pol_Sv = p['k'] * Pol * Sv 
    UB_Pol_Sv = p['k']*K['dG_pol'] * Sv_Pol
    B_Pol_Sc = p['k'] * Pol * Sc 
    UB_Pol_Sc = p['k']*K['dG_pol']  * Sc_Pol
    B_Np_Sv = p['k'] * NP * Sv_Pol 
    UB_Np_Sv = p['k'] * K['dG_np'] * Sv_Pol_Np
    B_Np_Sc = p['k'] * NP * Sc_Pol 
    UB_Np_Sc = p['k'] * K['dG_np'] * Sc_Pol_Np
    B_MCRNA_Pol = p['k'] * MCRNA * Pol 
    UB_MCRNA_Pol = p['k'] * K['dG_cRNA'] * MCRNA_Pol
    B_MVRNA_Pol = p['k'] * MVRNA * Pol 
    UB_MVRNA_Pol = p['k'] * K['dG_vRNA'] * MVRNA_Pol

    B_MCRNA_Pol_Pol = p['k'] * MCRNA_Pol * Pol
    UB_MCRNA_Pol_Pol = p['k'] * K['dG_cRNA'] * MCRNA_Pol_Pol
    B_MVRNA_Pol_Pol = p['k'] * MVRNA_Pol * Pol
    UB_MVRNA_Pol_Pol = p['k'] * K['dG_vRNA'] * MVRNA_Pol_Pol

    # Double stranded mini formation from capped MVRNA/MCRNA
    B_DS_MVRNA = p['k'] * cappedMCRNA * cappedMVRNA
    # dG_DF is to high to model -> irreversible
    UB_DS_MVRNA = 0 # p['k'] * K['dG_DS'] * dsMVRNA * 0.0

    # Compute transcription rates
    T_Pol = p['kt_pol'] * Sv * cap/(cap + K['dG_cap']) 
    T_NP = p['kt_np'] * Sv * cap/(cap + K['dG_cap']) 
    T_NEP = p['kt_nep'] * Sv * cap/(cap + K['dG_cap'])

    # Transcription of Minis into capped minis 
    # MCRNA_Pol -E-> MCRNA_c
    T_MCRNA = p['kt_mini'] * MCRNA_Pol * cap/(cap + K['dG_cap']) 
    # MVRNA_Pol -E-> MVRNA_c
    T_MVRNA = p['kt_mini'] * MVRNA_Pol * cap/(cap + K['dG_cap'])


    # Compute translation rates
    TL_Pol= p['ktl'] * mRNA_Pol
    TL_NP= p['ktl'] * mRNA_NP
    TL_NEP = p['ktl'] * mRNA_NEP

    # Compute degradation rates
    DEG_mRNA_Pol = p['kd'] * mRNA_Pol
    DEG_mRNA_NP = p['kd'] * mRNA_NP
    DEG_mRNA_NEP = p['kd'] * mRNA_NEP

    DEG_Pol = p['kdp'] * Pol
    DEG_NP = p['kdp'] * NP
    DEG_NEP = p['kdp'] * NEP

    DEG_mCRNA = p['kd_vc'] * MCRNA
    DEG_mVRNA = p['kd_vc'] * MVRNA

    DEG_cappedMCRNA = p['kd'] * cappedMCRNA
    DEG_cappedMVRNA = p['kd'] * cappedMVRNA

    # Export rates of Sv to the cytoplasm (is gated by NEP protein)
    # Sv -E-> Sv_c
    E_Sv = p['kexp'] * Sv * NEP 

    # RNP degradation
    # Sv, Sv -DEG-> 0
    DEG_Sv = p['kd_rnp'] * Sv
    DEG_Sc = p['kd_rnp'] * Sc

    # Cap dynamics
    # -VCAP-> CAP
    V_CAP = p['Vcap']

    # CAP -DEG-> 0
    DEG_CAP = p['kdcap'] * cap

    # Compute reaction rates

    rates = np.array([  R_Sc, R_Sv, R_I_MCRNA, R_MCRNA, R_MVRNA,
                        B_Pol_Sv, UB_Pol_Sv, B_Pol_Sc, UB_Pol_Sc, B_Np_Sv, UB_Np_Sv, B_Np_Sc, UB_Np_Sc, 
                        B_MCRNA_Pol, UB_MCRNA_Pol, B_MVRNA_Pol, UB_MVRNA_Pol,
                        T_Pol, T_NP, TL_Pol, TL_NP, DEG_mRNA_Pol, DEG_mRNA_NP, DEG_Pol, DEG_NP, DEG_mCRNA, DEG_mVRNA, E_Sv,
                        V_CAP, DEG_CAP, B_MCRNA_Pol_Pol, UB_MCRNA_Pol_Pol, B_MVRNA_Pol_Pol, UB_MVRNA_Pol_Pol, B_DS_MVRNA, UB_DS_MVRNA, 
                        T_NEP, TL_NEP, DEG_mRNA_NEP, DEG_NEP,
                        T_MCRNA, T_MVRNA, DEG_cappedMCRNA, DEG_cappedMVRNA, 
                        DEG_Sv, DEG_Sc]
                        )    

    
    # Compute reaction fluxes
    dxdt = S @ rates
    return dxdt


def build_reaction_stoichiometry(p):
    # Species to index mapping
    species = ['cap', 'Sv', 'Sc', 'Sv_Pol', 'Sc_Pol', 'Sv_Pol_Np', 'Sc_Pol_Np', 
                'MCRNA', 'MVRNA', 'MCRNA_Pol_Pol', 'MVRNA_Pol_Pol', 'mRNA_Pol', 
                'mRNA_NP', 'Pol', 'NP', 'Sv_c', 'MVRNA_Pol', 'MCRNA_Pol',
                'cappedMVRNA', 'cappedMCRNA', 'dsMVRNA', 'mRNA_NEP', 'NEP']
    species = {s: i for i, s in enumerate(species)}

    # Map reaciton rates to indices
    rates = ['R_Sc', 'R_Sv', 'R_I_MCRNA', 'R_MCRNA', 'R_MVRNA',
            'B_Pol_Sv', 'UB_Pol_Sv', 'B_Pol_Sc', 'UB_Pol_Sc', 'B_Np_Sv', 
            'UB_Np_Sv', 'B_Np_Sc', 'UB_Np_Sc',
            'B_MCRNA_Pol', 'UB_MCRNA_Pol', 'B_MVRNA_Pol', 'UB_MVRNA_Pol',
            'T_Pol', 'T_NP', 'TL_Pol', 'TL_NP', 'DEG_mRNA_Pol', 
            'DEG_mRNA_NP', 'DEG_Pol', 'DEG_NP', 'DEG_mCRNA', 'DEG_mVRNA', 'E_Sv',
            'V_CAP', 'DEG_CAP', 'B_MCRNA_Pol_Pol', 'UB_MCRNA_Pol_Pol', 'B_MVRNA_Pol_Pol', 
            'UB_MVRNA_Pol_Pol', 'B_DS_MVRNA', 'UB_DS_MVRNA',
            'T_NEP', 'TL_NEP', 'DEG_mRNA_NEP', 'DEG_NEP',
            'T_MCRNA', 'T_MVRNA', 'DEG_cappedMCRNA', 'DEG_cappedMVRNA', 
            'DEG_Sv', 'DEG_Sc']
    
    rates = {r: i for i, r in enumerate(rates)}

    # Build a matrix of reaction 
    M = np.zeros((len(rates), len(species)))

    # Replication rates
    # Sv_Pol_NP -R_Sc-> Sv_Pol_NP + Sc - z * NP
    M[rates['R_Sc'], species['NP']] = - p['z']
    M[rates['R_Sc'], species['Sc']] = 1

    # Sc_Pol_NP -R_Sv-> Sc_Pol_NP + Sv - z * NP
    M[rates['R_Sv'], species['NP']] = - p['z']
    M[rates['R_Sv'], species['Sv']] = 1

    # Initiation of mini replication
    # Sv_Pol -R_I_MCRNA-> MCRNA + Sv_Pol
    M[rates['R_I_MCRNA'], species['MCRNA']] = 1

    # Mini self replication
    # MVRNA_Pol -R_MCRNA-> MVRNA_Pol + MCRNA
    M[rates['R_MCRNA'], species['MCRNA']] = 1

    # MCRNA_Pol -R_MVRNA-> MCRNA_Pol + MVRNA
    M[rates['R_MVRNA'], species['MVRNA']] = 1

    # Binding rates
    # Pol + Sv -B_Pol_Sv-> Sv_Pol
    M[rates['B_Pol_Sv'], species['Pol']] = -1
    M[rates['B_Pol_Sv'], species['Sv']] = -1
    M[rates['B_Pol_Sv'], species['Sv_Pol']] = 1
    # Sv_Pol -UB_Pol_Sv-> Pol + Sv
    M[rates['UB_Pol_Sv'], species['Sv_Pol']] = -1
    M[rates['UB_Pol_Sv'], species['Pol']] = 1
    M[rates['UB_Pol_Sv'], species['Sv']]= 1

    # Pol + Sc -B_Pol_Sc-> Sc_Pol
    M[rates['B_Pol_Sc'], species['Pol']] = -1
    M[rates['B_Pol_Sc'], species['Sc']] = -1
    M[rates['B_Pol_Sc'], species['Sc_Pol']] = 1
    # Sc_Pol -UB_Pol_Sc-> Pol + Sc
    M[rates['UB_Pol_Sc'], species['Sc_Pol']] = -1
    M[rates['UB_Pol_Sc'], species['Pol']] = 1
    M[rates['UB_Pol_Sc'], species['Sc']]= 1

    # NP + Sv_Pol -B_Np_Sv-> Sv_Pol_Np
    M[rates['B_Np_Sv'], species['NP']] = -1
    M[rates['B_Np_Sv'], species['Sv_Pol']] = -1
    M[rates['B_Np_Sv'], species['Sv_Pol_Np']] = 1
    # Sv_Pol_Np -UB_Np_Sv-> NP + Sv_Pol
    M[rates['UB_Np_Sv'], species['Sv_Pol_Np']] = -1
    M[rates['UB_Np_Sv'], species['NP']]= 1
    M[rates['UB_Np_Sv'], species['Sv_Pol']] = 1

    # NP + Sc_Pol -B_Np_Sc-> Sc_Pol_Np
    M[rates['B_Np_Sc'], species['NP']] = -1
    M[rates['B_Np_Sc'], species['Sc_Pol']] = -1
    M[rates['B_Np_Sc'], species['Sc_Pol_Np']] = 1
    # Sc_Pol_Np -UB_Np_Sc-> NP + Sc_Pol
    M[rates['UB_Np_Sc'], species['Sc_Pol_Np']] = -1
    M[rates['UB_Np_Sc'], species['NP']]= 1
    M[rates['UB_Np_Sc'], species['Sc_Pol']] = 1

    # MCRNA + Pol -B_MCRNA_Pol-> MCRNA_Pol
    M[rates['B_MCRNA_Pol'], species['MCRNA']] = -1
    M[rates['B_MCRNA_Pol'], species['Pol']] = -1
    M[rates['B_MCRNA_Pol'], species['MCRNA_Pol']] = 1
    # MCRNA_Pol -UB_MCRNA_Pol-> MCRNA + Pol
    M[rates['UB_MCRNA_Pol'], species['MCRNA_Pol']] = -1
    M[rates['UB_MCRNA_Pol'], species['MCRNA']] = 1
    M[rates['UB_MCRNA_Pol'], species['Pol']] = 1

    # MVRNA + Pol -B_MVRNA_Pol-> MVRNA_Pol
    M[rates['B_MVRNA_Pol'], species['MVRNA']] = -1
    M[rates['B_MVRNA_Pol'], species['Pol']] = -1
    M[rates['B_MVRNA_Pol'], species['MVRNA_Pol']] = 1
    # MVRNA_Pol -UB_MVRNA_Pol-> MVRNA + Pol
    M[rates['UB_MVRNA_Pol'], species['MVRNA_Pol']] = -1
    M[rates['UB_MVRNA_Pol'], species['MVRNA']] = 1
    M[rates['UB_MVRNA_Pol'], species['Pol']] = 1

    # MCRNA_Pol -B_MCRNA_Pol_Pol-> MCRNA_Pol_Pol
    M[rates['B_MCRNA_Pol_Pol'], species['MCRNA_Pol']] = -1
    M[rates['B_MCRNA_Pol_Pol'], species['Pol']] = -1
    M[rates['B_MCRNA_Pol_Pol'], species['MCRNA_Pol_Pol']] = 1
    
    # MCRNA_Pol_Pol -UB_MCRNA_Pol_Pol-> MCRNA_Pol + Pol
    M[rates['UB_MCRNA_Pol_Pol'], species['MCRNA_Pol_Pol']] = -1
    M[rates['UB_MCRNA_Pol_Pol'], species['MCRNA_Pol']] = 1
    M[rates['UB_MCRNA_Pol_Pol'], species['Pol']] = 1

    # MVRNA_Pol -B_MVRNA_Pol_Pol-> MVRNA_Pol_Pol
    M[rates['B_MVRNA_Pol_Pol'], species['MVRNA_Pol']] = -1
    M[rates['B_MVRNA_Pol_Pol'], species['Pol']] = -1
    M[rates['B_MVRNA_Pol_Pol'], species['MVRNA_Pol_Pol']] = 1

    # MVRNA_Pol_Pol -UB_MVRNA_Pol_Pol-> MVRNA_Pol + Pol
    M[rates['UB_MVRNA_Pol_Pol'], species['MVRNA_Pol_Pol']] = -1
    M[rates['UB_MVRNA_Pol_Pol'], species['MVRNA_Pol']] = 1
    M[rates['UB_MVRNA_Pol_Pol'], species['Pol']] = 1

    # cappedMCRNA + cappedMVRNA -B_DS_MVRNA-> dsMVRNA
    M[rates['B_DS_MVRNA'], species['cappedMCRNA']] = -1
    M[rates['B_DS_MVRNA'], species['cappedMVRNA']] = -1
    M[rates['B_DS_MVRNA'], species['dsMVRNA']] = 1

    # dsMVRNA -UB_DS_MVRNA-> cappedMCRNA + cappedMVRNA
    M[rates['UB_DS_MVRNA'], species['dsMVRNA']] = -1
    M[rates['UB_DS_MVRNA'], species['cappedMCRNA']] = 1
    M[rates['UB_DS_MVRNA'], species['cappedMVRNA']] = 1
    
    # Transcription rates
    # Sv + Cap -T_Pol-> mRNA_Pol + Sv
    M[rates['T_Pol'], species['cap']] = -1
    M[rates['T_Pol'], species['mRNA_Pol']] = 1

    # Sv + Cap -T_NP-> mRNA_NP + Sv
    M[rates['T_NP'], species['cap']] = -1
    M[rates['T_NP'], species['mRNA_NP']] = 1

    # Sc + Cap -T_NEP-> mRNA_NEP + Sc
    M[rates['T_NEP'], species['cap']] = -1
    M[rates['T_NEP'], species['mRNA_NEP']] = 1

    # Transcription of Minis into capped minis
    # MCRNA + Cap -T_MCRNA-> cappedMCRNA + MCRNA
    M[rates['T_MCRNA'], species['cap']] = -1
    M[rates['T_MCRNA'], species['cappedMCRNA']] = 1

    # MVRNA + Cap -T_MVRNA-> cappedMVRNA + MVRNA
    M[rates['T_MVRNA'], species['cap']] = -1
    M[rates['T_MVRNA'], species['cappedMVRNA']] = 1

    # Translation rates
    # mRNA_Pol -TL_Pol-> mRNA_Pol + Pol
    M[rates['TL_Pol'], species['Pol']] = 1

    # mRNA_NP -TL_NP-> mRNA_NP + NP
    M[rates['TL_NP'], species['NP']] = 1

    # mRNA_NEP -TL_NEP-> mRNA_NEP + NEP
    M[rates['TL_NEP'], species['NEP']] = 1

    # Degradation rates
    # mRNA_Pol -DEG_mRNA_Pol-> 0
    M[rates['DEG_mRNA_Pol'], species['mRNA_Pol']] = -1

    # mRNA_NP -DEG_mRNA_NP-> 0
    M[rates['DEG_mRNA_NP'], species['mRNA_NP']] = -1
    
    # mRNA_NEP -DEG_mRNA_NEP-> 0
    M[rates['DEG_mRNA_NEP'], species['mRNA_NEP']] = -1

    # Pol -DEG_Pol-> 0
    M[rates['DEG_Pol'], species['Pol']] = -1

    # NP -DEG_NP-> 0
    M[rates['DEG_NP'], species['NP']] = -1

    # NEP -DEG_NEP-> 0
    M[rates['DEG_NEP'], species['NEP']] = -1

    # MCRNA -DEG_mCRNA-> 0
    M[rates['DEG_mCRNA'], species['MCRNA']] = -1

    # MVRNA -DEG_mVRNA-> 0
    M[rates['DEG_mVRNA'], species['MVRNA']] = -1

    # cappedMCRNA -DEG_cappedMCRNA-> 0
    M[rates['DEG_cappedMCRNA'], species['cappedMCRNA']] = -1

    # cappedMVRNA -DEG_cappedMVRNA-> 0
    M[rates['DEG_cappedMVRNA'], species['cappedMVRNA']] = -1
    
    # Sv -DEG_Sv-> 0
    M[rates['DEG_Sv'], species['Sv']] = -1

    # Sc -DEG_Sc-> 0
    M[rates['DEG_Sc'], species['Sc']] = -1

    # Export rates
    # Sv -E_Sv-> Sv + Sv_c
    M[rates['E_Sv'], species['Sv']] = -1
    M[rates['E_Sv'], species['Sv_c']] = 1

    # Cap dynamics
    # -V_CAP-> CAP
    M[rates['V_CAP'], species['cap']] = 1

    # CAP -DEG_CAP-> 0
    M[rates['DEG_CAP'], species['cap']] = -1

    return M


def default_parameters(V=V):
    p = OrderedDict()

    # [1] Parameter values from https://journals.asm.org/doi/10.1128/jvi.00080-12#F3
    
    # Binding processes
    # This is instant compared to the other processes so we can set it to a high value
    p['k'] = 1e9 * 3600/ (NA * V) # Diffusion limited rate constant in [1e9 1/s/M] 

    # Transciption processes
    kt = 35 # NEP Transcription rate constant [1/h]
    p['kt_pol'] = 0.5 * kt # Transcription rate constant [1/h]
    p['kt_np'] = 3 * kt # Transcription rate constant [1/h]
    p['kt_nep'] = 1 * kt # Transcription rate constant [1/h]
    p['kt_mini'] = 0.1 * kt # Transcription rate constant [1/h]

    # Import into the nucleus is the rate limiting step in this case 
    # Translation rate about 64,800 nt h−1 vs mRNA synthesis 2.5 10−5 nt h−1
    p['ktl'] = 6.0 # [1] mRNA -> nuclear protein [1/h] this includes translation and transport out and back into the nucleus

    # Replication processes
    p['krv'] = 13.86 # [1] vRNA Replication rate constant [1/h]
    p['krc'] = 1.38 # [1] cRNA Replication rate constant [1/h]
    
    # Mini viral RNA replication rate
    p['krm'] = p['krc']* 0.1 # Rate at with minivial RNA is initiated << than cRNA replication rate

    # Degradation processes
    p['kd'] = 0.33 # [1] RNA Degradation rate constant [1/h]
    p['kd_vc'] = 36 # [1] nascent cvRNA Degradation rate constant [1/h]
    p['kd_rnp'] = 0.09 # [1] RNP Degradation rate constant [1/h]
    p['kdp'] = 0.5 # [1] Protein Degradation rate constant [1/h] ??

    # Nuclear export of segments into the cytoplasm
    p['kexp'] = 1e-6 * 100.0 # [1] Export rate constant [1/h/#NEP molecules] (adopted)

    # Stoichimetry of POL to NP
    p['z'] = 20 # [CI3E] Stoichiometry of NP to POL

    # Host RNA dynamics / This is an estimation of the host RNA dynamics in vicinity of the 
    # viral replication machinery
    # 
    p['Vcap'] = 1e3 # Cap synthesis rate
    p['kdcap'] = 0.1 # Cap degradation rate

    # Put citations from AJ here 
    # Cap binding
    p['dG_cap'] = np.log(1e-6)*R*T0 # Cap binding to PA

    p['dG_vRNA'] = np.log(0.5e-9)*R*T0 # Free energy of binding of MVRNA to Pol 
    p['dG_cRNA'] = np.log(13e-9)*R*T0 # Free energy of binding of MC to Pol 

    p['dG_pol'] = np.log(100e-9)*R*T0  # Free energy of binding of pol/pol binding
    p['dG_np'] = np.log(15e-9)*R*T0  # Free energy of binding of NP to mrna/pol complex

    # Temperature
    p['T'] = T0

    # KM for NP binding to activate -> 1 molecule 
    # p['K_Np'] = 1

    return p


def binding_constants(dG,T,V=V):
    """
    dG: np.array of free energies in kJ/mol
    RT: float, temperature in Kelvin times the gas constant

    Notes 50 kJ/mol ~ nM binding constant
    """
    K = np.exp(dG/R/T) * NA * V # * 1 mol/L -> convert to number of molecules
    return K
