@echo off
:: Define number of loops for each sample value
set loopscartpole=1   :: number of loops when samps=3
set loops2=20    :: number of loops when samps=5
set loopsLunar=10    :: number of loops when samps=5
set loopsAcrobot=4


set envThang="acrobot"
echo running %loopscartpole%
for /L %%i in (1,1,%loopsAcrobot%) do (
python relevancyGetter.py --config_env %envThang% --norm 0 --samps=250 --num=%%i
)
set envThang="cartpole"
echo running %envThang%
for /L %%i in (1,1,%loopscartpole%) do (
python relevancyGetter.py --config_env %envThang% --norm 1 --samps=1000 --num=%%i
)
set envThang="_bipedal"
echo running %envThang%
for /L %%i in (1,1,%loops2%) do (
python relevancyGetter.py --config_env %envThang% --norm 1 --samps=50 --num=%%i
)
set envThang="_lunar_lander"
echo running %envThang%
for /L %%i in (1,1,%loopsLunar%) do (
python relevancyGetter.py --config_env %envThang% --norm 1 --samps=100 --num=%%i
)
set envThang="acrobot"
echo running %envThang%
for /L %%i in (1,1,%loopsAcrobot%) do (
python relevancyGetter.py --config_env %envThang% --norm 1 --samps=250 --num=%%i
)
:: Loop for samps=5
