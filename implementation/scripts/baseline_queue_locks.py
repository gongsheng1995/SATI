"""Prepared queue lock handoff for bounded baseline supervisors and leaves.

An inherited descriptor refers to the same open file description as its parent,
so a child verifies/reuses that flock instead of opening a conflicting lock.
No GPU work, model loading or process launch occurs in this module.
"""
import os


def acquire_queue_locks(paths,inherited_spec=None):
    import fcntl
    descriptors=None
    if inherited_spec is not None:
        values=inherited_spec.split(',')
        if len(values)!=len(paths) or any(not value.isdecimal() for value in values):
            raise ValueError('Exactly one inherited lock descriptor per expected queue is required.')
        descriptors=[int(value) for value in values]
        if len(set(descriptors))!=len(descriptors) or any(fd<3 for fd in descriptors):
            raise ValueError('Invalid/duplicated inherited queue descriptors.')
    locks=[]
    try:
        for index,path in enumerate(paths):
            path.parent.mkdir(parents=True,exist_ok=True)
            if descriptors is None:
                lock=path.open('a')
            else:
                fd=descriptors[index]
                actual=os.fstat(fd);expected=path.stat()
                if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
                    raise ValueError('Inherited descriptor does not refer to the expected queue lock.')
                lock=os.fdopen(os.dup(fd),'a')
            locks.append(lock)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return locks
    except BaseException:
        for lock in locks:lock.close()
        raise
