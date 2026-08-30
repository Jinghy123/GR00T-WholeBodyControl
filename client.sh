# Usage: bash client.sh [run_name]
# run_name names the recording folder under zed_recordings/ (default: timestamp).
# The ZED recorder subscribes to the server's viewer PUB stream (5559), which is
# separate from the inference REP socket (5558), so recording never blocks the
# policy client.

RUN_NAME="${1:-recordings}"
OUT_DIR="zed_recordings/${RUN_NAME}"
echo "[client.sh] recording into $OUT_DIR"

# Pin the sonic env's python so a stray venv on PATH (e.g. .venv_sim) can't
# shadow it — `python` alone picks up whatever the terminal inherited.
PY="$HOME/miniforge3/envs/sonic/bin/python"
# IMAGE_FIT=pad    -> pad ZED 672x376 to 672x384 with black rows at the bottom (matches LeRobot conversion)
# IMAGE_FIT=resize -> stretch to 672x384 (legacy, default)
IMAGE_FIT="${IMAGE_FIT:-pad}"
ZED_SERVER="${ZED_SERVER:-192.168.123.164}"

"$PY" zed_recorder.py --server "$ZED_SERVER" --out-dir "$OUT_DIR" &
REC_PID=$!

cleanup() {
    if kill -0 "$REC_PID" 2>/dev/null; then
        kill -INT "$REC_PID" 2>/dev/null
        wait "$REC_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# TASK_PROMPT="pick up the gray hippo toy and place it into the orange bowl"
# TASK_PROMPT="pick up the green grapes and place it into the green bowl"
# TASK_PROMPT="pick up the banana and place it into the wooden box"
# TASK_PROMPT="pick up the eggplant and place it into the transparent box"
# TASK_PROMPT="hold the dustpan and sweep the white paper scraps into it with the brush"
# TASK_PROMPT="hold the dustpan and sweep the yellow and green bottle caps into it with the brush"
# TASK_PROMPT="hold the dustpan and sweep the black plastic pieces into it with the brush"
# TASK_PROMPT="grasp the backrest of the chair and push it straight under the table"
# TASK_PROMPT="grasp the backrest of the chair, turn left, and push it under the table"
# TASK_PROMPT="grasp the backrest of the chair, turn right, and push it under the table"
# TASK_PROMPT="pick up the foil bag and turn left and throw it into the trash can"
# TASK_PROMPT="pick up the foil bag and turn right and throw it into the trash can"
# TASK_PROMPT="pick up the red snack box and turn left and throw it into the trash can"
# TASK_PROMPT="pick up the red snack box and turn right and throw it into the trash can"
# TASK_PROMPT="pick up the paper ball and turn left and throw it into the trash can"
# TASK_PROMPT="pick up the paper ball and turn right and throw it into the trash can"
# TASK_PROMPT="pick up the snack bag and turn left and throw it into the trash can"
# TASK_PROMPT="pick up the snack bag and turn right and throw it into the trash can"
# TASK_PROMPT="grasp the yellow bottle, open the top drawer of the kitchen island, place the bottle inside, and close the drawer"
# TASK_PROMPT="grasp the transparent bottle, open the top drawer of the kitchen island, place the bottle inside, and close the drawer"
# TASK_PROMPT="grasp the silver can, open the top drawer of the kitchen island, place the can inside, and close the drawer"
# TASK_PROMPT="grasp the green drink bottle, open the top drawer of the kitchen island, place the bottle inside, and close the drawer"
# TASK_PROMPT="open the top-right door of the cabinet, grab the purple flower, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="open the top-left door of the cabinet, grab the white flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="open the top-left door of the cabinet, grab the pink flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="open the top-left door of the cabinet, grab the purple flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="open the top-left door of the cabinet, grab the orange flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="gather up the yellow shirt and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the gray shirt and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the white shirt and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the black trousers and yellow shirt and turn right and put them into the laundry basket"
# TASK_PROMPT="gather up the yellow shirt and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the yellow shirt with right hand and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the black trousers and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the yellow shirt and gray shirt and turn right and put them into the laundry basket"
# TASK_PROMPT="gather up the gray shirt with right hand and turn right and put it into the laundry basket"
# TASK_PROMPT="open the top-right door of the cabinet, grab the orange flower, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="open the top-right door of the cabinet, grab the white flower, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="open the top-right door of the cabinet, grab the pink flower, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="pick up the foil bag and turn right and throw it into the trash can"
# TASK_PROMPT="pick up the red snack box and turn right and throw it into the trash can"
# TASK_PROMPT="grasp the transparent water bottle and pour the water into the yellow kettle"
# TASK_PROMPT="hold the pink watering can by the handle and water the white flower"
# TASK_PROMPT="pick up the snack bag and turn left and throw it into the trash can"
# TASK_PROMPT="kneel down and scoop up the beige pillow near the bed and put it at the head of the bed"
# TASK_PROMPT="open the top-right door of the cabinet, grab the purple flower, turn left, and place it at the top-right corner of the table"
# TASK_PROMPT="kneel down, hook the beige shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_PROMPT="kneel down, hook the purple shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_PROMPT="kneel down, hook the white shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_PROMPT="kneel down, hook the blue shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_PROMPT="kneel down and scoop up the white pillow near the bed and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the beige pillow near the bed and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white pillow near the kitchen island and turn left and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the beige pillow near the kitchen island and turn left and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white pillow near the shoe rack and turn around and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the beige pillow near the shoe rack and turn around and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white pillow near the laundry basket and turn around and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the beige pillow near the laundry basket and turn around and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white pillow near the kitchen island and turn left and put it at the head of the bed"


# L1 prompts
# TASK_PROMPT="kneel down, hook the pink shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_PROMPT="kneel down, hook the white shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_PROMPT="gather up the yellow shirt and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the gray shirt and turn right and put it into the laundry basket"
# TASK_PROMPT="gather up the black trousers and turn right and put them into the laundry basket"
# TASK_PROMPT="gather up the white shirt and turn right and put it into the laundry basket"
# TASK_PROMPT="grasp the backrest of the chair and push it straight under the table"
# TASK_PROMPT="grasp the backrest of the chair, turn left, and push it under the table"
# TASK_PROMPT="grasp the backrest of the chair, turn right, and push it under the table"
# TASK_PROMPT="kneel down and scoop up the white toy near the shoe rack and turn around and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the green pillow near the bed and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white toy near the bed and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the green pillow near the kitchen island, turn left, and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white toy near the kitchen island, turn left, and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the green pillow near the shoe rack, turn around, and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white toy near the shoe rack, turn around, and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the green pillow near the laundry basket, turn around, and put it at the head of the bed"
# TASK_PROMPT="kneel down and scoop up the white toy near the laundry basket, turn around, and put it at the head of the bed"

# L2 prompts
# TASK_PROMPT="grasp the chip box and place it into the basket"
# TASK_PROMPT="pick up the banana, open the top drawer of the kitchen island, place the ball inside, and close the drawer"
# TASK_PROMPT="kneel down, gather up the yellow shirts, turn around and put it into the laundry basket"
# TASK_PROMPT="gather up the white shirt with right hand and turn right and throw it into the trash can"
# TASK_PROMPT="pick up the yellow shirt with right hand and turn right and throw it into the trash can"
# TASK_PROMPT="pick up the banana and turn right and throw it into the trash can"
# TASK_PROMPT="gather up the gray hippo toy with right hand and turn right and throw it into the trash can"
# TASK_PROMPT="open the top-right door of the cabinet, pick up the tennis ball, turn left, and place it in the basket"
# TASK_PROMPT="open the top-left door of the cabinet, pick up the white dumpling, close the cabinet door, turn left, and place it in the basket"
# TASK_PROMPT="open the top-left door of the cabinet, gather up the blue shirt with right hand, close the cabinet door, turn left, and put it into the laundry basket"
# TASK_PROMPT="open the top-left door of the cabinet, grasp the green drink bottle, close the cabinet door, turn left, and throw it into the trash can"
# TASK_PROMPT="hold the block and turn left and throw it into the trash can"
# TASK_PROMPT="kneel down, pick up the banana, turn around and throw it into the trash can"
# TASK_PROMPT="hook the white shoes and turn left and throw it into the trash can"
# TASK_PROMPT="open the top-left door of the cabinet, grab the pink flower, close the cabinet door, turn left, and throw it into the trash can"
# TASK_PROMPT="kneel down, hook the white shoes on the first tier of the shoe rack, turn around, and put it into the laundry basket"
TASK_PROMPT="kneel down, hook the white shoes on the first tier of the shoe rack, turn around, and throw it into the trash can"

"$PY" g1_sonic_client_rtc.py \
    --policy-host 127.0.0.1 --policy-port 5000 \
    --prompt "$TASK_PROMPT" \
    --execution-horizon 12 --inference-delay 10 --guidance-weight 5.0 --kv-scheme stride1 --include-neck \
    --image-fit "$IMAGE_FIT"