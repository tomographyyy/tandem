"""
Copyright 2024 Tomohiro TAKAGAWA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from mpi4py import MPI
import sys

class DecoratorMPI(object):
    def finalize(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if MPI.COMM_WORLD.rank==0:
                    print ("-----------")
                    print ("Exception:", str(e), flush=True)
                    print ("-----------")
                    import traceback
                    import sys
                    import time
                    print ("-----------")
                    traceback.print_exc(file=sys.stdout)
                    print ("-----------")
                    
                    sys.stdout.flush()
                    sys.stderr.flush()
                else:
                    time.sleep(1)
                MPI.COMM_WORLD.Abort(1)
                MPI.Finalize()
        return wrapper
