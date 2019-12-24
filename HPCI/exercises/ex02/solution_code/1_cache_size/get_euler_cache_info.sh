#!/bin/bash

COMMAND='lscpu && echo ----------------------------- && grep . /sys/devices/system/cpu/cpu0/cache/index*/*'

bsub -R "select[model==XeonGold_6150]" -R fullnode -n 36 -W 00:01 "echo Euler IV XeonGold_6150 && $COMMAND"
bsub -R "select[model==XeonE3_1585Lv5]"            -n 4  -W 00:01 "echo Euler III XeonE3_1585Lv5 && $COMMAND"
bsub -R "select[model==XeonE5_2680v3]" -R fullnode -n 24 -W 00:01 "echo Euler II XeonE5_2680v3 && $COMMAND"
