import time
import pyspacemouse


def main():
    device = pyspacemouse.open()
    print("open result:", device)

    print("Move the SpaceMouse. Press Ctrl+C to stop.")

    try:
        while True:
            state = device.read()

            if state is None:
                print("No state")
                time.sleep(0.1)
                continue

            print(
                f"x={state.x:+.4f}, y={state.y:+.4f}, z={state.z:+.4f}, "
                f"roll={state.roll:+.4f}, pitch={state.pitch:+.4f}, yaw={state.yaw:+.4f}, "
                f"buttons={state.buttons}"
            )

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()