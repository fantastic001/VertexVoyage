import pickle
import os

def stateful(cls):
    class Wrapper:
        mycls = cls 
        def __init__(self, obj, path) -> None:
            self.obj = obj
            self.path = path
        def __getattr__(self, attr):
            x = getattr(self.obj, attr)
            if callable(x) and attr != "publisher":
                def W(*args, **kw):
                    result = x(*args, **kw)
                    self.save()
                    return result
                return W
            else:
                return x
        def save(self):
            pickle.dump(self.obj, open(self.path, "wb"))
    cls.load = lambda path: Wrapper(pickle.load(open(path, "rb")), path) if os.path.exists(path) else Wrapper(cls(), path)
    return cls