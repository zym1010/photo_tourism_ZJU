import os

img_dir = '/data/MM21/zhongwen/guochuang/'

for root, dirnames, filenames in os.walk(img_dir):
   for filename in filenames:
      (name, ext) = os.path.splitext(filename)
      if ext != '.jpg':
        continue
      new_dir = os.path.join(root,  name)

      os.mkdir(new_dir)
      os.system('mv {0} {1}'.format(os.path.join(root, filename), new_dir))

      gz_filename = os.path.join(root,  name + '.tar.gz')
      os.system('tar zcvf {0} {1}'.format(gz_filename, new_dir))
