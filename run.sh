python g1_sonic_client_rtc.py \
    --policy-host 127.0.0.1 --policy-port 5000 \
    --prompt "pick up the green grapes and place it into the green bowl" \
    --execution-horizon 12 --inference-delay 10 --guidance-weight 5.0 --kv-scheme stride1