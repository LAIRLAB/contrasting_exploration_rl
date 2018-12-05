#!/bin/bash

echo "NEEDS TO BE EXECUTED IN THE PROJECT ROOT"

echo "RUNNING ARS ON LQR"
echo "TUNING ARS"
python -m ars.test.tune_lqr_ars
echo "RUNNING ARS"
python -m ars.test.run_lqr_ars

echo "RUNNING ExAct ON LQR"
echo "TUNING ExAct"
python -m exact.test.tune_lqr_exact
echo "RUNNING ExAct"
python -m exact.test.run_lqr_exact
