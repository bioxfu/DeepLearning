from PIL import Image
import lmdb
import caffe
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

def read_from_lmdb(lmdb_path, img_save_to, prefix):
    lmdb_env = lmdb.open(lmdb_path, map_size=3221225472)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    datum_index = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = datum.data
        channel = datum.channels
        print('Datum channels: %d' % datum.channels)
        print('Datum width: %d' % datum.width)
        print('Datum height: %d' % datum.height)
        print('Datum data length: %d' % len(datum.data))
        print('Datum label: %d' % datum.label)

        size = datum.width * datum.height
        pixles1 = datum.data[0:size]
        pixles2 = datum.data[size:2*size]
        pixles3 = datum.data[2*size:3*size]
        #Extract images of different channel
        image1 = Image.frombytes('L', (datum.width, datum.height), pixles1)
        image2 = Image.frombytes('L', (datum.width, datum.height), pixles2)
        image3 = Image.frombytes('L', (datum.width, datum.height), pixles3)
        #注意三通道的顺序，如果LMDB中图像是按照BGR存储的则需要按照：image3,image2,image1的顺序合并为RGB图像。PIL中图像是按照RGB的顺序存储的
        image4 = Image.merge("RGB",(image3,image2,image1))
        image4.save(img_save_to+"/"+str(label)+"/"+prefix+"_"+str(key)+".jpg")
        datum_index += 1
        print("extracted")
    lmdb_env.close()

read_from_lmdb('../datasets/root_shoot/root/lmdb/42_testing/',  '../datasets/root_shoot/root/image/', 'test')
read_from_lmdb('../datasets/root_shoot/root/lmdb/42_training/', '../datasets/root_shoot/root/image/', 'train')

read_from_lmdb('../datasets/root_shoot/shoot/lmdb/wheat-large_testing/',  '../datasets/root_shoot/shoot/image/', 'test')
read_from_lmdb('../datasets/root_shoot/shoot/lmdb/wheat-large_training/', '../datasets/root_shoot/shoot/image/', 'train')
