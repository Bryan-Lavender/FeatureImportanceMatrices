
@echo off
:: Define number of loops for each sample value
set loops1=1   :: number of loops when samps=3
set loops2=1    :: number of loops when samps=5
set envThang="cartpole"
echo Running RunnAllDaTrees.py with --samps=3 --config_env %envThang%

set envThang="_half_cheetah"
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 3 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=5
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 5 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=10
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 10 --config_env %envThang%
)
:: Loop for samps=5
echo Running RunnAllDaTrees.py with --samps=20
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 20 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=40
for /L %%i in (1,1,%loops1%) do (
    python RunnAllDaTrees.py --samps 40 --config_env %envThang%
)

:: Loop for samps=5
echo Running RunnAllDaTrees.py with --samps=60
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 60 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=100
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 100 --config_env %envThang%
)



set envThang="acrobot"
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 3 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=5
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 5 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=10
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 10 --config_env %envThang%
)
:: Loop for samps=5
echo Running RunnAllDaTrees.py with --samps=20
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 20 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=40
for /L %%i in (1,1,%loops1%) do (
    python RunnAllDaTrees.py --samps 40 --config_env %envThang%
)

:: Loop for samps=5
echo Running RunnAllDaTrees.py with --samps=60
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 60 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=100
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 100 --config_env %envThang%
)


set envThang="_lunar_lander"
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 3 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=5
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 5 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=10
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 10 --config_env %envThang%
)
:: Loop for samps=5
echo Running RunnAllDaTrees.py with --samps=20
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 20 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=40
for /L %%i in (1,1,%loops1%) do (
    python RunnAllDaTrees.py --samps 40 --config_env %envThang%
)

:: Loop for samps=5
echo Running RunnAllDaTrees.py with --samps=60
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 60 --config_env %envThang%
)

echo Running RunnAllDaTrees.py with --samps=100
for /L %%i in (1,1,%loops2%) do (
    python RunnAllDaTrees.py --samps 100 --config_env %envThang%
)
echo All runs complete!