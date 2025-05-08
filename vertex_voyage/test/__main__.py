import unittest
import os 

class DebugTestResult(unittest.TextTestResult):
    def __init__(self, *args, failfast=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.failfast = failfast
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.failfast:
            # Raise the exception to trigger debugger
            raise err[1].with_traceback(err[2])

    def addError(self, test, err):
        super().addError(test, err)
        if self.failfast:
            # Raise the exception to trigger debugger
            raise err[1].with_traceback(err[2])


if __name__ == '__main__':
    failfast = os.getenv('FAILFAST', '0') == '1'
    loader = unittest.TestLoader()
    start_dir = 'vertex_voyage/test'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner(resultclass=DebugTestResult, failfast=failfast)
    runner.run(suite)
