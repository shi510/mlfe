import wgetter, gzip, tarfile

fileName = wgetter.download('https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz')
gzip_file = gzip.open('cifar-10-binary.tar.gz', 'rb')
ungzip_file = open('cifar-10-binary.tar', 'wb')
ungzip_file.write(gzip_file.read())
gzip_file.close()
ungzip_file.close()

tar_file = tarfile.open('cifar-10-binary.tar', 'r')
tar_file.extractall('.')
tar_file.close()
