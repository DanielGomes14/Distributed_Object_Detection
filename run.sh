#!/bin/bash
DEFAULT_NUMBER_WORKERS=2

workers=${1:-$DEFAULT_NUMBER_WORKERS}

source venv/bin/activate

python3 server.py &
sleep 4



for i in $(seq 1 $workers); do
  celery -A worker worker --loglevel=info --concurrency=1 -n warkaaa$i.%h  & sleep 1
done

curl -F 'video=@moliceiro.m4v' http://localhost:5000
