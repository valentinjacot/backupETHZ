# run Ncandles 1..3 and repeat each 10 times
for N in {1..3}
do


for i in {1..10}
do
    bsub -n 1 -J task2a -o 2a_N${N}_${i}.out -W 4:00 ./task2a ${N} 100000
    sleep 30s # make sure jobs start with different seed
done #experiments


done #nCandles
