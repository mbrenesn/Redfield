#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/d/dvira/mbrenes/slurm-%j.out
#SBATCH -p compute
#SBATCH -J test

cd $SLURM_SUBMIT_DIR

module load intel/2020u4

#for thet in $(seq 0.0 0.03927 3.1416)
#do
#    for phi in $(seq 0.0 0.03927 3.1416)
#    do
#      ./Redfield.x --rc 6 --la_l 20.0 --la_r 20.0 --t_l 2.0 --t_r 1.0 --theta_l $thet --theta_r $phi >> /scratch/d/dvira/mbrenes/NonCommutingBaths/RC6la20.0dT1.0HeatPi.dat
#    done
#done
#for i in 0.1 0.10896528 0.11873432 0.12937918 0.14097839 0.15361749 0.16738973 0.18239669 0.19874906 0.21656747 0.23598335 0.25713991 0.28019322 0.30531333 0.33268552 0.3625117 0.39501189 0.43042581 0.46901468 0.51106316 0.5568814 0.60680737 0.66120935 0.72048861 0.78508242 0.85546725 0.93216228 1.01573323 1.10679655 1.20602395 1.31414736 1.43196434 1.56034394 1.70023313 1.85266378 2.01876025 2.19974775 2.39696127 2.61185554 2.84601568 3.10116893 3.37919738 3.68215185 4.01226704 4.37197798 4.76393801 5.19103835 5.65642943 6.16354411 6.71612304 7.31824222 7.97434306 8.68926517 9.46828205 10.31713996 11.24210035 12.24998602 13.34823146 14.54493767 15.84893192 
#do
#  ./Redfield.x --rc 6 --la_l $i --la_r $i --t_l 3.0 --t_r 1.0 --theta_l 2.3562 --theta_r 0.7854 >> /scratch/d/dvira/mbrenes/NonCommutingBaths/RC6dT2.0HeatTheta3pi4Phipi4.dat
#done
for i in $(seq 0.0 0.1 30.0)
do
  ./RedfieldThetaCurr.x --rc 8 --la_l $i --la_r $i --t_l 1.0 --t_r 0.5 --theta_l 1.57079633 --theta_r 1.57079633 >> /scratch/d/dvira/mbrenes/NonCommutingBaths/EffTL1.0TR0.5HeatThetapi2Phipi2.dat
done
