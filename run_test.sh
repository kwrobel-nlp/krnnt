#!/usr/bin/env bash

#check if server is running
SERVER_STARTED=0
if [ `ps aux | grep krnnt_serve -c` -eq 1 ]; then
    echo 'Starting server'
    ./start_flask_server.sh > /dev/null 2>&1 &
    PID=$!
    echo "PID: $PID"
    SERVER_STARTED=1
    sleep 5
fi


cd tests
python3 -m pytest


if [ $SERVER_STARTED -eq 1 ]; then
    echo 'Killing server'
    pkill -P "$PID"
fi