#!/bin/bash

echo "NEEDS TO BE EXECUTED IN THE PROJECT ROOT"

echo "RUNNING ARS ON SWIMMER"
echo "TUNING ARS"
python -m ars.test.tune_mujoco_ars --env_name='Swimmer-v2'
echo "RUNNING ARS"
python -m ars.test.run_mujoco_ars --env_name='Swimmer-v2'

echo "RUNNING ARS ON HALFCHEETAH"
echo "TUNING ARS"
python -m ars.test.tune_mujoco_ars --env_name='HalfCheetah-v2'
echo "RUNNING ARS"
python -m ars.test.run_mujoco_ars --env_name='HalfCheetah-v2'

echo "RUNNING ExAct ON SWIMMER"
echo "TUNING ExAct"
python -m exact.test.tune_mujoco_exact --env_name='Swimmer-v2'
echo "RUNNING ExAct"
python -m exact.test.run_mujoco_exact --env_name='Swimmer-v2'

echo "RUNNING ExAct ON HALFCHEETAH"
echo "TUNING ExAct"
python -m exact.test.tune_mujoco_exact --env_name='HalfCheetah-v2'
echo "RUNNING ExAct"
python -m exact.test.run_mujoco_exact --env_name='HalfCheetah-v2'

