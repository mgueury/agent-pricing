pip3.9 install -r requirements.txt

sudo firewall-cmd --zone=public --add-port=8081/tcp --permanent
sudo firewall-cmd --reload

sudo update-alternatives --config python3
# Choice 2