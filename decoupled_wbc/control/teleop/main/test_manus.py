#!/usr/bin/env python3
"""Real-time Wuji hand visualization using OpenCV."""

import time
import cv2
import numpy as np
import zmq
import msgpack


# Correct joint names for Wuji hand (26 joints)
LEFT_JOINT_NAMES = [
    'LeftHandWrist', 'LeftHandPalm',
    'LeftHandThumbMetacarpal', 'LeftHandThumbProximal', 'LeftHandThumbDistal', 'LeftHandThumbTip',
    'LeftHandIndexMetacarpal', 'LeftHandIndexProximal', 'LeftHandIndexIntermediate', 'LeftHandIndexDistal', 'LeftHandIndexTip',
    'LeftHandMiddleMetacarpal', 'LeftHandMiddleProximal', 'LeftHandMiddleIntermediate', 'LeftHandMiddleDistal', 'LeftHandMiddleTip',
    'LeftHandRingMetacarpal', 'LeftHandRingProximal', 'LeftHandRingIntermediate', 'LeftHandRingDistal', 'LeftHandRingTip',
    'LeftHandLittleMetacarpal', 'LeftHandLittleProximal', 'LeftHandLittleIntermediate', 'LeftHandLittleDistal', 'LeftHandLittleTip',
]

RIGHT_JOINT_NAMES = [name.replace('Left', 'Right') for name in LEFT_JOINT_NAMES]

# Skeleton connections (indices)
CONNECTIONS = [
    (0, 1),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]

FINGER_COLORS = [
    (255, 0, 255),    # Thumb
    (0, 255, 255),    # Index
    (255, 255, 0),    # Middle
    (0, 165, 255),    # Ring
    (0, 0, 255),      # Little
]


def get_positions(hand_data, joint_names):
    positions = []
    for name in joint_names:
        if name in hand_data:
            positions.append(hand_data[name][0])
        else:
            positions.append([0, 0, 0])
    return np.array(positions)


def project_3d_to_2d(positions, scale=600, center=(320, 240)):
    projected = []
    cx, cy = center
    for pos in positions:
        x = int(cx + pos[0] * scale)
        y = int(cy - pos[2] * scale)
        projected.append((x, y))
    return projected


def draw_hand(img, positions, color_base, offset_x=0):
    if len(positions) == 0:
        return

    points_2d = project_3d_to_2d(positions, center=(320 + offset_x, 240))

    # Draw connections first (thinner)
    for start, end in CONNECTIONS:
        if start < len(points_2d) and end < len(points_2d):
            if start >= 2:
                finger_idx = min((start - 2) // 5, 4)
                color = FINGER_COLORS[finger_idx]
            else:
                color = color_base

            cv2.line(img, points_2d[start], points_2d[end], color, 2)

    # Draw joints (smaller)
    for i, pt in enumerate(points_2d):
        if i in [5, 10, 15, 20, 25]:  # Fingertips
            cv2.circle(img, pt, 3, (255, 255, 255), -1)
            cv2.circle(img, pt, 3, color_base, 1)
        else:  # Other joints
            cv2.circle(img, pt, 2, color_base, -1)


def main():
    print("==> Wuji Hand Real-time Visualization")
    print("==> Press 'q' to exit")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect("tcp://localhost:5559")
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, 50)

    img_size = (640, 480)

    # Store last valid data to prevent flickering
    last_left_pos = None
    last_right_pos = None

    frame_count = 0
    start_time = time.time()
    last_print = time.time()
    fps = 0

    try:
        while True:
            img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

            # Try to get new data
            data_received = False
            try:
                msg = socket.recv(flags=zmq.NOBLOCK)
                unpacker = msgpack.Unpacker(raw=False)
                unpacker.feed(msg)

                for data in unpacker:
                    if isinstance(data, dict) and 'left' in data:
                        left_pos = get_positions(data['left'], LEFT_JOINT_NAMES)
                        right_pos = get_positions(data['right'], RIGHT_JOINT_NAMES)

                        # Update if data is valid
                        if not np.allclose(left_pos, 0, atol=0.001):
                            last_left_pos = left_pos
                        if not np.allclose(right_pos, 0, atol=0.001):
                            last_right_pos = right_pos

                        data_received = True
                        frame_count += 1
                        break

            except zmq.Again:
                pass

            # Draw with last valid data
            if last_left_pos is not None:
                draw_hand(img, last_left_pos, (0, 0, 255), offset_x=-160)
            if last_right_pos is not None:
                draw_hand(img, last_right_pos, (255, 0, 0), offset_x=160)

            # FPS
            if time.time() - last_print >= 0.5:
                fps = frame_count / (time.time() - start_time)
                last_print = time.time()

            cv2.putText(img, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "Left: Red | Right: Blue", (10, img_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Wuji Hand Tracking", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
