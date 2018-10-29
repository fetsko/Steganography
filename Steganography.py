#! /usr/local/bin/python3.4


import numpy as np
import time
import base64
import scipy
from scipy import misc
from PIL import Image
from io import StringIO
import zlib
import re

class Payload:

    def __init__(self, img=None, compressionLevel=-1, content=None):
        if (((img is None) & (content is None)) | ((compressionLevel < -1) | (compressionLevel > 9))):
            raise ValueError

        if (content is None) & (type(img).__module__ != np.__name__):
            raise TypeError
        elif ((img is None) & (type(content).__module__ != np.__name__)):
            raise TypeError

        if ((img is not None) & (compressionLevel is not None)):
            self.img = img
            if len(img.shape) == 2:
                self.type = 'Gray'
            else:
                self.type = 'Color'
            if (compressionLevel != -1):
                self.compress(compressionLevel)
                self.compressed = True
            else:
                self.compress(-1)
                self.compressed = False

            self.xml = self.xml_serialization(self.compressed_data, self.type, [self.dim1, self.dim2], self.compressed)
            self.content = self.base64encode(self.xml)

        else:
            self.content = content
            data = self.base64decode(self.content)
            x = self.xml_deserialization(data)

            to_be_decomp = x[0]
            try:
                zlib.decompress(to_be_decomp)
            except:
                decompressed_data = bytes(to_be_decomp)
                self.compressed = False
            else:
                decompressed_data = zlib.decompress(to_be_decomp)
                self.compressed = True

            decompressed_data = np.fromstring(decompressed_data, dtype=np.uint8)

            shape = x[2]
            self.type = x[1]
            if (self.type == 'Color'):
                raw_data = np.empty(tuple((shape[0], shape[1], 3)))

                raw_data[:, :, 0] = decompressed_data[0:shape[0] * shape[1]].reshape(shape, order='C')
                raw_data[:, :, 1] = decompressed_data[shape[0] * shape[1]:2 * shape[0] * shape[1]].reshape(shape, order='C')
                raw_data[:, :, 2] = decompressed_data[2 * shape[0] * shape[1]:3 * shape[0] * shape[1]].reshape(shape, order='C')
                self.img = raw_data.astype(dtype='uint8')

            else:
                raw_data = np.empty(tuple((shape[0], shape[1])))
                raw_data = decompressed_data[0:shape[0] * shape[1]].reshape(shape, order='C')

    def test_func(self,x):
        if ((x >= 65) & (x <= 90)):
            return x - 65
        elif ((x >= 97) & (x <= 122)):
            return x - 71
        elif ((x >= 48) & (x <= 57)):
            return x + 4
        elif (x == 43):
            return x + 19
        elif (x == 47):
            return x + 16
        else:
            return x

    def test_func2(self,x):
        if ((x >= 0) & (x <= 25)):
            return chr(x + 65)
        elif ((x >= 26) & (x <= 51)):
            return chr(x + 71)
        elif ((x >= 52) & (x <= 61)):
            return chr(x - 4)
        elif (x == 62):
            return chr(x - 19)
        elif (x == 63):
            return chr(x - 16)
        else:
            return chr(x)

    def compress(self, level):
        self.dim1 = self.img.shape[0]
        self.dim2 = self.img.shape[1]

        if (self.type == 'Color'):
            self.size = (self.img.shape[0] * self.img.shape[1] * 3)

            raw_data = np.empty(self.size)

            raw_data = self.img[:, :, 0].flatten('C')
            raw_data = np.append(raw_data, self.img[:, :, 1].flatten('C'))
            raw_data = np.append(raw_data, self.img[:, :, 2].flatten('C'))
        else:
            self.size = (self.img.shape[0] * self.img.shape[1])
            raw_data = np.empty(self.size)
            raw_data = self.img.flatten('C')

        if level == -1:
            self.compressed_data = bytes(raw_data)
        else:
            self.compressed_data = zlib.compress(raw_data,level)

    def xml_serialization(self, data, type, size, compressed):
        xml = '<?xml version="1.0" encoding="UTF-8"?><payload type="' + type + '" size="' + str(size[0]) + ',' + str(
                size[1]) + '" compressed="' + str(compressed) + '">'

        data = str(list(data)).replace(' ', '').replace('[', '').replace(']', '')
        xml = xml + data + '</payload>'

        return xml

    def xml_deserialization(self, data):

        image_type = re.findall(r'<payload type="(.*?)"', data)
        image_type = image_type[0]
        size1 = re.findall(r' size="(.*?),', data)
        size2 = re.findall(r' size="\w*,(.*?)"', data)

        size = tuple((int(size1[0]), int(size2[0])))

        compressed = re.findall(r' compressed="(.*?)"', data)
        compressed = bool(compressed[0])

        data = re.findall(r' compressed="\w*">(.*?)</payload>', data)

        data = data[0].split(',')
        data = list(map(int, data))
        data = bytearray(data)

        return tuple((data, image_type, size, compressed))

    def base64encode(self, xml_string):

        xml_string = base64.b64encode(bytes(xml_string, 'utf-8'))

        temp = str(xml_string.decode())
        temp = np.fromstring(temp,dtype=np.uint8)

        vfunc = np.vectorize(self.test_func,otypes=[np.uint8])
        temp2 = vfunc(temp)

        if (temp2[-2] == 61):
            temp3 = temp2[:-2]
            return temp3
        elif (temp2[-1] == 61):
            temp3 = temp2[:-1]
            return temp3

        return temp2

    def base64decode(self,data):

        vfunc = np.vectorize(self.test_func2,otypes=[np.str])
        temp = vfunc(data)

        #s = StringIO()

        #np.savetxt(s,temp,fmt='%s')
        #print(s.getvalue())
        temp2 = ''.join(temp)

        data = str(base64.b64decode(temp2 + '=' * (-len(temp2) % 4)))

        return data

class Carrier:

    def __init__(self, img):
        if type(img).__module__ != np.__name__:
            raise TypeError

        self.img = img
        if len(img.shape) == 2:
            self.type = 'Gray'
        else:
            self.type = 'Color'

    def test_func2(self,x):
        if ((x >= 0) & (x <= 25)):
            return chr(x + 65)
        elif ((x >= 26) & (x <= 51)):
            return chr(x + 71)
        elif ((x >= 52) & (x <= 61)):
            return chr(x - 4)
        elif (x == 62):
            return chr(x - 19)
        elif (x == 63):
            return chr(x - 16)
        else:
            return chr(x)

    def payloadExists(self):

        header = np.array([15,  3, 61, 56, 27, 22, 48, 32, 29, 38, 21, 50, 28, 54, 37, 47, 27, 35, 52, 34 ,12, 18, 56, 48,  8, 34,  1, 37, 27, 38, 13, 47, 25,  6, 37, 46, 25, 51, 52, 34, 21,
                             21, 17,  6, 11, 19, 32, 34, 15, 51, 56, 60, 28,  6,  5, 57, 27,  6, 61, 33, 25,  2,  1, 52, 30, 23,  1, 37, 15, 18])
        if self.type == 'Color':
            red_data = self.img[:, :, 0].flatten('C')
            green_data = self.img[:, :, 1].flatten('C')
            blue_data = self.img[:, :, 2].flatten('C')

            extracted_payload = np.empty_like(red_data[0:70])
            extracted_payload = ((red_data[0:70] & 3)) + ((green_data[0:70] & 3) << 2) + ((blue_data[0:70] & 3) << 4)
            extracted_payload = np.array(extracted_payload)

            #extracted_xml = self.base64decode(extracted_payload)
        else:
            bw_data = self.img[:,:].flatten('C')

            extracted_payload = np.empty_like([bw_data[0:70]])
            extracted_payload = ((bw_data[:210:3] & 3)) + ((bw_data[1:210:3] & 3) << 2) + ((bw_data[2:210:3] & 3) << 4)
            extracted_payload = np.array(extracted_payload)

            #extracted_xml = self.base64decode(extracted_payload)

        #print(extracted_xml)
        if ((header == extracted_payload).all()):
            return True
        else:
            return False
        #if (re.findall(r'<payload type="(.*?)"', extracted_xml)):
        #    return True
        #else:
        #    return False

    def clean(self):
        temp = np.random.randint(4, size=self.img.shape)
        temp = (self.img & 252) + temp

        return temp

    def embedPayload(self, payload, override=False):

        if (self.type == 'Color'):
            max_size = (self.img.shape[0] * self.img.shape[1])
        else:
            max_size = int((self.img.shape[0] * self.img.shape[1])/3)

        if (isinstance(payload,Payload) is False):
            raise TypeError('Parameter passed must be of type Payload')
        elif (len(payload.content) > max_size):
            raise ValueError('Payload size is larger than what carrier can hold')

        if (override is False):
            if self.payloadExists():
                raise Exception('Current carrier already contains a payload, and override is False')

        first_2 = np.empty_like(payload.content)
        mid_2 = np.empty_like(payload.content)
        last_2 = np.empty_like(payload.content)

        first_2 = (payload.content >> 4) & 3
        mid_2 = (payload.content >> 2) & 3
        last_2 = (payload.content) & 3

        shape = self.img.shape

        size = (self.img.shape[0] * self.img.shape[1] * 3)

        if (self.type == 'Color'):

            raw_data2 = np.empty(size)

            raw_data2 = self.img[:, :, 0].flatten('C')
            raw_data2 = np.append(raw_data2, self.img[:, :, 1].flatten('C'))
            raw_data2 = np.append(raw_data2, self.img[:, :, 2].flatten('C'))

            payload_size = first_2.shape[0]

            self.payloadSize = payload_size


            raw_data2[0:payload_size] = (raw_data2[0:payload_size] & 252) + last_2
            raw_data2[(shape[0] * shape[1]):(shape[0] * shape[1] + payload_size)] = (raw_data2[(shape[0] * shape[1]):(shape[0] * shape[1] + payload_size)] & 252) + mid_2
            raw_data2[(2 * shape[0] * shape[1]):(2 * shape[0] * shape[1] + payload_size)] = (raw_data2[(2 * shape[0] * shape[1]):(2 * shape[0] * shape[1] + payload_size)] & 252) + first_2

            raw_data3 = np.empty(shape)
            raw_data3 = raw_data3.astype(dtype='uint8')

            raw_data3[:,:,0] = raw_data2[0:shape[0] * shape[1]].reshape(shape[0:2], order='C')
            raw_data3[:,:,1] = raw_data2[shape[0] * shape[1]:shape[0] * shape[1] * 2].reshape(shape[0:2], order='C')
            raw_data3[:,:,2] = raw_data2[shape[0] * shape[1] * 2:shape[0] * shape[1] * 3].reshape(shape[0:2], order='C')

        else:
            raw_data2 = np.empty(size)

            raw_data2 = self.img[:, :].flatten('C')
            payload_size = first_2.shape[0]

            self.payloadSize = payload_size

            raw_data2[:payload_size*3:3] = (raw_data2[:payload_size*3:3] & 252) + last_2
            raw_data2[1:payload_size*3:3] = (raw_data2[1:payload_size*3:3] & 252) + mid_2
            raw_data2[2:payload_size*3:3] = (raw_data2[2:payload_size*3:3] & 252) + first_2

            raw_data3 = np.empty(shape)
            raw_data3 = raw_data3.astype(dtype='uint8')

            raw_data3 = raw_data2.reshape(shape, order='C')

        return raw_data3


    def extractPayload(self):
        if (self.type == 'Color'):

            size = (self.img.shape[0] * self.img.shape[1] * 3)

            red_data = self.img[:, :, 0].flatten('C')
            green_data = self.img[:, :, 1].flatten('C')
            blue_data = self.img[:, :, 2].flatten('C')

            extracted_header = np.empty_like(115)
            extracted_header = ((red_data[0:115] & 3)) + ((green_data[0:115] & 3) << 2) + ((blue_data[0:115] & 3) << 4)
            header_xml = self.base64decode(extracted_header)
            size1 = re.findall(r' size="(.*?),', header_xml)
            size2 = re.findall(r' size="\w*,(.*?)"', header_xml)

            pay_size = tuple((int(size1[0]), int(size2[0])))

            data_size = pay_size[0]*pay_size[1]*15
            if (data_size > red_data.shape[0]):
                data_size = red_data.shape[0]

            extracted_payload = np.empty_like(data_size)
            extracted_payload = ((red_data[0:data_size] & 3)) + ((green_data[0:data_size] & 3) << 2) + ((blue_data[0:data_size] & 3) << 4)

            extracted_xml = self.base64decode(extracted_payload)


            x = self.unpack(extracted_xml)

            to_be_decomp = x[0]

            try:
                zlib.decompress(to_be_decomp)
            except:
                decompressed_data = bytes(to_be_decomp)
            else:
                decompressed_data = zlib.decompress(to_be_decomp)

            decompressed_data = np.fromstring(decompressed_data, dtype=np.uint8)

            shape = x[2]

            if (x[1] == 'Color'):
                raw_data = np.empty(tuple((shape[0],shape[1],3)))

                raw_data[:,:,0] = decompressed_data[0:shape[0]*shape[1]].reshape(shape,order='C')
                raw_data[:,:,1] = decompressed_data[shape[0]*shape[1]:2*shape[0]*shape[1]].reshape(shape,order='C')
                raw_data[:,:,2] = decompressed_data[2*shape[0]*shape[1]:3*shape[0]*shape[1]].reshape(shape,order='C')
            else:
                raw_data = np.empty(tuple((shape[0], shape[1])))
                raw_data = decompressed_data.reshape(shape,order='C')
        else:
            size = (self.img.shape[0] * self.img.shape[1])

            bw_data = self.img[:, :].flatten('C')

            extracted_header = np.empty_like(115)
            extracted_header = ((bw_data[0:115*3:3] & 3)) + ((bw_data[1:115*3:3] & 3) << 2) + ((bw_data[2:115*3:3] & 3) << 4)
            header_xml = self.base64decode(extracted_header)
            size1 = re.findall(r' size="(.*?),', header_xml)
            size2 = re.findall(r' size="\w*,(.*?)"', header_xml)

            pay_size = tuple((int(size1[0]), int(size2[0])))

            data_size = pay_size[0]*pay_size[1]*15

            if (data_size*3 > bw_data.shape[0]):
                data_size = int(bw_data.shape[0]/3)

            extracted_payload = np.empty_like(data_size)

            extracted_payload = ((bw_data[0:data_size*3:3] & 3)) + ((bw_data[1:data_size*3:3] & 3) << 2) + ((bw_data[2:data_size*3:3] & 3) << 4)

            #print('decode start')
            #start = time.clock()
            extracted_xml = self.base64decode(extracted_payload)
            #end = time.clock()
            #print(end-start)


            x = self.unpack(extracted_xml)

            to_be_decomp = x[0]

            try:
                zlib.decompress(to_be_decomp)
            except:
                decompressed_data = bytes(to_be_decomp)
            else:
                decompressed_data = zlib.decompress(to_be_decomp)

            decompressed_data = np.fromstring(decompressed_data, dtype=np.uint8)

            shape = x[2]

            if (x[1] == 'Color'):
                raw_data = np.empty(tuple((shape[0],shape[1],3)))

                raw_data[:,:,0] = decompressed_data[0:shape[0]*shape[1]].reshape(shape,order='C')
                raw_data[:,:,1] = decompressed_data[shape[0]*shape[1]:2*shape[0]*shape[1]].reshape(shape,order='C')
                raw_data[:,:,2] = decompressed_data[2*shape[0]*shape[1]:3*shape[0]*shape[1]].reshape(shape,order='C')
            else:
                raw_data = np.empty(tuple((shape[0], shape[1])))
                raw_data = decompressed_data.reshape(shape,order='C')

        return Payload(img=raw_data.astype(dtype='uint8'))


    def base64decode(self,data):
        vfunc = np.vectorize(self.test_func2,otypes=[np.str])
        temp = vfunc(data)

        #s = StringIO()

        #np.savetxt(s,temp,fmt='%s')
        #print(s.getvalue())
        temp2 = ''.join(temp)

        data = str(base64.b64decode(temp2 + '=' * (-len(temp2) % 4)))

        return data


    def unpack(self, data):
        image_type = re.findall(r'<payload type="(.*?)"', data)

        image_type = image_type[0]
        size1 = re.findall(r' size="(.*?),', data)
        size2 = re.findall(r' size="\w*,(.*?)"', data)

        size = tuple((int(size1[0]), int(size2[0])))

        compressed = re.findall(r' compressed="(.*?)"', data)
        compressed = bool(compressed[0])


        data = re.findall(r' compressed="\w*">(.*?)</payload>', data)

        data = data[0].split(',')
        data = list(map(int, data))
        data = bytearray(data)

        return tuple((data, image_type, size, compressed))


if __name__ == "__main__":
    #yoshi = misc.yoshi()
    #misc.imsave('Yoshi.png',yoshi)


    #x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #x[2::3] = 11
    #print(x)

    image1 = misc.imread('extra_data/payload4.png')
    #print(image1.shape)

    #print(len(image1.shape))

    #image1 = misc.imread('red_panda2.png')
    #image2 = misc.imread('earth.png')
    image2 = misc.imread('extra_data/carrier3.png')
    #image3 = misc.imread('data/payload2.png')
    #image3 = misc.imread('husky.png')

    payload1 = Payload(img=image1,compressionLevel=5)
    #payload1 = Payload(img=image1)

    #payload2 = Payload(content=payload1.content)

    #payload3 = Payload(img=image3)

    #print((payload1.img == payload2.img).all())
    carrier1 = Carrier(image2)
    #cleaned = carrier1.clean()

    a = carrier1.embedPayload(payload1,True)
    #b = carrier1.embedPayload(payload3,True)

    carrier2 = Carrier(a)

    payload2 = carrier2.extractPayload().img
    #carrier3 = Carrier(b)

    #print("main")
    print(carrier1.payloadExists())
    print(carrier2.payloadExists())
    #print(carrier3.payloadExists())

    #payload2 = carrier2.extractPayload()



    misc.imsave('blk_white.png', payload2)
    #misc.imsave('red_panda2_test.png', payload2.img)
