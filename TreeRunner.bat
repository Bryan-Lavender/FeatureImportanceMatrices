@echo off
:: Define number of loops for each sample value
set loops1=1   :: number of loops when samps=3
set loops2=1    :: number of loops when samps=5


set envThang="_bipedal"

:: Loop for samps=5
echo Running RunnAllDaTrees.py with --samps=60
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 60 --config_env %envThang% --norm 0
)

