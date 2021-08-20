import wgetter, gzip

file_names = [
                'train-images-idx3-ubyte', 
                'train-labels-idx1-ubyte', 
                't10k-images-idx3-ubyte', 
                't10k-labels-idx1-ubyte'
            ]

for i in range(0, len(file_names)):
    fileName = wgetter.download('https://storage.googleapis.com/cvdf-datasets/mnist/%s.gz' % file_names[i])
    inF = gzip.open(file_names[i] + '.gz', 'rb')
    outF = open(file_names[i], 'wb')
    outF.write(inF.read())
    inF.close()
    outF.close()
