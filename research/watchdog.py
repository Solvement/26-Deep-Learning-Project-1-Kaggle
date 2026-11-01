"""Stop only the exact owned child if its supervisor disappears or budget ends."""
import argparse
import time
import psutil


def matching_process(pid, created):
    try:
        process = psutil.Process(pid)
        return process if abs(process.create_time()-created) < .01 else None
    except psutil.NoSuchProcess:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--parent-created", type=float, required=True)
    parser.add_argument("--child", type=int, required=True)
    parser.add_argument("--child-created", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    deadline = time.monotonic()+args.seconds
    while True:
        child = matching_process(args.child, args.child_created)
        if child is None:
            return
        if matching_process(args.parent, args.parent_created) is None or time.monotonic() >= deadline:
            child.kill()
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
