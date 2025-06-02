@echo off

set envThang = "cartpole"
set loops2 = 1
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 3 --config_env %envThang%
)