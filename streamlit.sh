#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
export PATH=~/.local/bin/:$PATH

. ./env.sh
# export TF_VAR_comparment_ocid=xxxxx
# export TF_VAR_db_password=xxxxx

streamlit run streamlit.py --server.port 8080 2>&1 | tee tools.log

# Ex: curl "http://$BASTION_IP:8080/"
