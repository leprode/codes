from torch.distributed.elastic.multiprocessing import start_processes

log_dir = "/tmp/test"

import os
import shutil
shutil.rmtree(log_dir)
os.makedirs(log_dir)

# caution; arguments casted to string, runs:
# echo "1" "2" "3" and echo "[1, 2, 3]"
start_processes(
   name="trainer",
   entrypoint="/bin/echo",
   args={0:("sasasasa",), 1:([1,2,3],)},
   envs={0:{}, 1:{}},
   log_dir=log_dir
 )
