
def f():
    try:
        raise queue.Empty
    except queue.Empty:
        pass


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.Process(target=f).start()
