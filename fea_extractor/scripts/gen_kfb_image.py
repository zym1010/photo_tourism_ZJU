"""
	./$cluster/ColorSIFTWrapper ./shared/colorDescriptor keyframes/$video.tar.gz tmp/"$feature"_raw_$video.tar.gz harrislaplace $desc $feature
        ./$cluster/TxycWrapper ./$cluster/txyc kmeans/$feature.center tmp/"$feature"_raw_$video.tar.gz tmp/"$feature"_txyc_$video.tar.gz $feature txyc 4096 10 $dim
        ./$cluster/SpbofWrapper ./$cluster/spbof tmp/"$feature"_txyc_$video.tar.gz tmp/"$feature"_spbof_$video.tar.gz txyc spbof 4096 10 1 $w $h
        ./$cluster/BofMerger tmp/"$feature"_spbof_$video.tar.gz tmp/"$feature"_merged_$video.spbof spbof
"""

import os
from PIL import Image
import sys

if len(sys.argv) != 4:
    print 'Usage: gen_cmd.py <fea_name> <tar_gz_list> dataset'
    sys.exit(1)

dims = {'sift':128, 'csift':384, 'tch':45}

feature = sys.argv[1]
video_list = sys.argv[2]
dataset = sys.argv[3]

dim = dims[feature]

if feature == 'tch':
    desc = 'transformedcolorhistogram'
else:
    desc = feature

bin_dir = '/home/zhongwen/sfep/ColorDescriptors/rocks/'
shared_dir = '/home/zhongwen/sfep/ColorDescriptors/shared/'
center = '/home/zhongwen/sfep/ColorDescriptors/kmeans/{feature}.center'.format(feature = feature)
kf_dir =    '/data/MM21/zhongwen/{0}/images'.format(dataset)
output_dir = '/data/MM21/zhongwen/{0}/features/{1}/'.format(dataset, feature)

(ColorSIFTWrapper, TxycWrapper, SpbofWrapper, BofMerger, txyc, spbof) = map(lambda x : os.path.join(bin_dir, x), ('ColorSIFTWrapper', 'TxycWrapper', 'SpbofWrapper', 'BofMerger', 'txyc', 'spbof'))
colorDescriptor = os.path.join(shared_dir, 'colorDescriptor')

with open(video_list) as f:
    for keyframe in f:
        keyframe = keyframe.strip()
        (kframe_dir, image_name) = os.path.split(keyframe)
        if kframe_dir[-1] != '/':
            kframe_dir = kframe_dir + '/'
        out_dir = kframe_dir.replace(kf_dir, output_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        gz = keyframe.replace('.tar.gz', '_raw.tar.gz')
        gz = gz.replace(kf_dir, output_dir)
        txyc_gz = gz.replace('_raw', '_txyc')
        spbof_gz = gz.replace('_raw', '_spbof')
        merged_spbof = gz.replace('_raw.tar.gz', '.spbof')
        record = gz.replace('_raw.tar.gz', '.record')
        name = image_name[0: -len('.tar.gz')]
        image_path = os.path.join(kframe_dir, name, name +'.jpg')
        try:
           im = Image.open(image_path) 
        except:
           pass
        (w, h) = im.size
        resolution = str(w)+' '+str(h)
        stdout_file = gz.replace('_raw.tar.gz', '.stdout')
        stderr_file = gz.replace('_raw.tar.gz', '.stderr')
        cmd1 = '{ColorSIFTWrapper} {colorDescriptor} {keyframe} {gz} harrislaplace {desc} {feature}'.format(ColorSIFTWrapper = ColorSIFTWrapper, colorDescriptor= colorDescriptor, keyframe = keyframe, gz = gz, desc = desc, feature = feature)
#        cmd2 = '{TxycWrapper} {txyc} {center} {gz} {txyc_gz} {feature} txyc 4096 10 {dim}'.format(TxycWrapper = TxycWrapper, txyc = txyc, center = center, gz = gz, txyc_gz = txyc_gz, feature = feature, dim = dim)
 #       cmd3 = '{SpbofWrapper} {spbof} {txyc_gz} {spbof_gz} txyc spbof 4096 10 1 {resolution}'.format(SpbofWrapper = SpbofWrapper, spbof = spbof, txyc_gz = txyc_gz, spbof_gz = spbof_gz, resolution = resolution)
  #      cmd4 = '{BofMerger} {spbof_gz} {merged_spbof} spbof'.format(BofMerger = BofMerger, spbof_gz = spbof_gz, merged_spbof = merged_spbof)
#        cmd5 = 'touch {record}'.format(record = record)
        cmd = '({0} > {1} 2>{2}'.format(cmd1, stdout_file, stderr_file)
        print cmd
f.close()
