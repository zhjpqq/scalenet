import os

import time

import zipfile

import tarfile


def jieya(source_dir, target_dir, fmt='zip'):
    '''
    将源文件夹下的所有压缩包，解压到目标文件夹下
    '''

    def untar(fname, dirs):
        f = tarfile.open(fname)
        f.extractall(path=dirs)
        f.close()

    def unzip(fname, dirs):
        f = zipfile.ZipFile(fname, 'r')
        f.extractall(path=dirs)
        f.close()

    if fmt == 'zip':
        unfunc = unzip
    elif fmt == 'tar':
        unfunc = untar
    else:
        raise NotImplementedError('未知的压缩格式，%s' % fmt)

    xtic = time.time()
    for parent, dir_names, file_names in os.walk(source_dir):
        for file_name in file_names:
            tic = time.time()
            print(file_name, '-------- start!')
            file_path = os.path.join(source_dir, file_name)
            print(file_path)
            unfunc(file_path, dirs=target_dir)
            print(file_name, '-------- done! -- in --- %.4f s \n'% (time.time()-tic, ))

    print('\nAll file has been jieya in %.4f s...\n' % (time.time()-xtic, ) )


if __name__ == "__main__":
    source_dir = '/data1/jpzhang/datasets/imagenet/train_zip'
    target_dir = '/data1/jpzhang/datasets/imagenet/train'

    jieya(source_dir, target_dir, fmt='zip')
