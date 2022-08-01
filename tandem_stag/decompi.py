from mpi4py import MPI

class DecoratorMPI(object):
    def finalize(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                print ("except", flush=True)
                import traceback
                import sys
                import time
                traceback.print_exc()
                sys.stderr.flush()
                time.sleep(0.1)
                MPI.COMM_WORLD.Abort(1)
                MPI.Finalize()
        return wrapper
