import sys
import os
import dircache
import string
import stat
import shutil

proto_dir = 'src/VulkanTest/'
target_dir = 'src/'

def make_example(src_dir, dest_dir, projname, depth):
  print(depth,"make", src_dir, dest_dir)
  try:
    os.mkdir(dest_dir)
  except:
    pass
  for fname in dircache.listdir(src_dir):
    mode = os.stat(src_dir + fname).st_mode
    if 'VulkanTest' in fname:
      dest_name = string.replace(fname, 'VulkanTest', projname)
    else:
      dest_name = fname
    print(depth, "fname", fname)
    if stat.S_ISDIR(mode):
      make_example(src_dir + fname + '/', dest_dir + '/' + dest_name + '/', projname, depth+"  ")
    else:
      print(depth, dest_dir + '/' + dest_name)
      try:
        test = open(dest_dir + '/' + dest_name, "rb")
        test.close()
      except:
        out = open(dest_dir + '/' + dest_name, "wb")
        for line in open(src_dir + fname):
          line = string.replace(line, 'VulkanTest', projname)
          out.write(line)

num_args = len(sys.argv)

if num_args == 1 or "--help" in sys.argv:
  print("\nmakes or updates a new octet project in src/")
  print("\nusage: generate_project.py projectname")
  exit()

for i in range(1,num_args):
  arg = sys.argv[i];
  if arg[0] != '-':
    make_example(proto_dir, target_dir + arg, arg, "")
  else:
    print("unrecognised option " + arg)
    exit()
