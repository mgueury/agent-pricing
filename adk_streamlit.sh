#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
export PATH=~/.local/bin/:$PATH

. ../app/env.sh
source $HOME/myenv/bin/activate

streamlit run adk_streamlit.py --server.port 8081 2>&1 | tee adk_streamlit.log

# Ex: curl "http://$BASTION_IP:8081/"
