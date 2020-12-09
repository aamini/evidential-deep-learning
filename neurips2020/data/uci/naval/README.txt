=================================================================================================================
Human Activity Recognition Using Smartphones Dataset
Version 1.0
=================================================================================================================
Andrea Coraddu(2), Luca Oneto(1), Alessandro Ghio(1), Stefano Savio(2), Davide Anguita(1), Massimo Figari(2)
1 - Smartlab - Non-Linear Complex Systems Laboratory
DITEN - Universitˆ  degli Studi di Genova, Genoa (I-16145), Italy. 
{luca.oneto,alessandro.ghio,davide.anguita}@unige.it
2 - Marine Technology Research Team
DITEN - Universitˆ  degli Studi di Genova, Genoa (I-16145), Italy. 
{andrea.coraddu,stefano.savio,massimo.figari}@unige.it
=================================================================================================================

The experiments have been carried out by means of a numerical simulator of a naval vessel (Frigate) characterized by a Gas Turbine (GT) propulsion plant. The different blocks forming the complete simulator (Propeller, Hull, GT, Gear Box and Controller) have been developed and fine tuned over the year on several similar real propulsion plants. In view of these observations the available data are in agreement with a possible real vessel.
In this release of the simulator it is also possible to take into account the performance decay over time of the GT components such as GT compressor and turbines.
The propulsion system behaviour has been described with this parameters:
- Ship speed (linear function of the lever position lp).
- Compressor degradation coefficient kMc.
- Turbine degradation coefficient kMt.
so that each possible degradation state can be described by a combination of this triple (lp,kMt,kMc).
The range of decay of compressor and turbine has been sampled with an uniform grid of precision 0.001 so to have a good granularity of representation.
In particular for the compressor decay state discretization the kMc coefficient has been investigated in the domain [1; 0.95], and the turbine coefficient in the domain [1; 0.975].
Ship speed has been investigated sampling the range of feasible speed from 3 knots to 27 knots with a granularity of representation equal to tree knots.
A series of measures (16 features) which indirectly represents of the state of the system subject to performance decay has been acquired and stored in the dataset over the parameter's space.
Check the README.txt file for further details about this dataset.

For each record it is provided:
======================================

- A 16-feature vector containing the GT measures at steady state of the physical asset:
    Lever position (lp) [ ]
    Ship speed (v) [knots]
    Gas Turbine (GT) shaft torque (GTT) [kN m]
    GT rate of revolutions (GTn) [rpm]
    Gas Generator rate of revolutions (GGn) [rpm]
    Starboard Propeller Torque (Ts) [kN]
    Port Propeller Torque (Tp) [kN]
    Hight Pressure (HP) Turbine exit temperature (T48) [C]
    GT Compressor inlet air temperature (T1) [C]
    GT Compressor outlet air temperature (T2) [C]
    HP Turbine exit pressure (P48) [bar]
    GT Compressor inlet air pressure (P1) [bar]
    GT Compressor outlet air pressure (P2) [bar]
    GT exhaust gas pressure (Pexh) [bar]
    Turbine Injecton Control (TIC) [%]
    Fuel flow (mf) [kg/s]
- GT Compressor decay state coefficient
- GT Turbine decay state coefficient

The dataset includes the following files:
=========================================

- 'README.txt'

- 'Features.txt': List of all features.

- 'data.txt': Dataset.

Notes: 
======
- Features are not normalized
- Each feature vector is a row on the text file (18 elements in each row)

For more information about this dataset please contact: cbm@smartlab.ws
Check at www.cbm.smartlab.ws for updates on this dataset.

License:
========
Use of this dataset in publications must be acknowledged by referencing the following publication [1] 

[1] A. Coraddu, L. Oneto, A. Ghio, S. Savio, D. Anguita, M. Figari, Machine Learning Approaches for Improving Condition?Based Maintenance of Naval Propulsion Plants, Journal of Engineering for the Maritime Environment, 2014, DOI: 10.1177/1475090214540874, (In Press)

@article{Coraddu2013Machine,
    author={Coraddu, Andrea and Oneto, Luca and Ghio, Alessandro and 
                 Savio, Stefano and Anguita, Davide and Figari, Massimo},
    title={Machine Learning Approaches for Improving Condition?Based Maintenance of Naval Propulsion Plants},
    journal={Journal of Engineering for the Maritime Environment},
    volume={--},
    number={--},
    pages={--},
    year={2014}
}

This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.

Other Related Publications:
===========================

[2] M. Altosole, G. Benvenuto, M. Figari, U. Campora, Real-time simulation of a cogag naval ship propulsion system, Proceedings of the Institution of Mechanical Engineers, Part M: Journal of Engineering for the Maritime Environment 223 (1) (2009) 47-62.

=================================================================================================================
Andrea Coraddu, Luca Oneto, Alessandro Ghio, Stefano Savio, Davide Anguita, Massimo Figari. September 2014.
