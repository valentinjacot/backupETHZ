# run Ncandles 1..3 and repeat each 10 times
for N in {1..3}
do


for i in {1..10}
do
    bsub -n 1 -J task2b -o 2b_N${N}_${i}.out -W 4:00 ./task2b ${N} 100000
    sleep 30s # make sure jobs start with different seed
done #experiments


done #nCandles
