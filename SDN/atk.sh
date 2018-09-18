#!/bin/bash

tcpdump -teni attacker-eth0 -w /home/uki/capture.pcap &
sleep 2
iperf -c 192.168.0.3 -t 2 -i 1 #hping3 disini
sleep 2
pid=$(ps aux | grep tcpdump | awk '{print $2}')
kill -9 $pid
sleep 2
tshark -r /home/uki/capture.pcap -T fields -E separator=, -E header=y -E quote=d -e _ws.col.Time -e _ws.col.Protocol -e _ws.col.Length > /home/uki/capture.csv