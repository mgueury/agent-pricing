#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
export PATH=~/.local/bin/:$PATH

. ../app/env.sh

streamlit run streamlit.py --server.port 8081 2>&1 | tee streamlit.log

# Ex: curl "http://$BASTION_IP:8081/"
