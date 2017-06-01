import multiprocessing as mp
import time


def worker(q, gpu_idx):
    print(mp.current_process().name, "working")
    while True:
        try:
            item = q.get(False, 3)
            print(mp.current_process().name, "got", item, "on gpu", gpu_idx)
            time.sleep(2)  # simulate a "long" operation
        finally:
            break


if __name__ == "__main__":
    the_queue = mp.Queue()
    ps = []
    for gpu in [0, 2]:
        p = mp.Process(target=worker, args=(the_queue, gpu))
        ps.append(p)
        p.start()

    the_queue.put(3)
    the_queue.put(1)
    the_queue.put(2)

    the_queue.close()
    the_queue.join_thread()

    #p1.join()
    #p2.join()
