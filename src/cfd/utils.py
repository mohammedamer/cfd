def log(msg):
    print(msg)


def log_tensor_stats(tensor):
    log(f"{tensor.max()=} {tensor.min()=}")
