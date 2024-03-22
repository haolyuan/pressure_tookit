import numpy as np
import tqdm


def progress_bar(producer, text=None, shuffle=False, verbose=True):
    if type(producer) is int:
        producer = range(producer)
    if shuffle:
        np.random.shuffle(producer)
    if not verbose:
        return producer
    return tqdm.tqdm(list(producer), ascii=True, desc=text, dynamic_ncols=True)
