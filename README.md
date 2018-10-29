# Steganography
Steganography.py embeds/extracts "payload" (smaller images) into/out of "carrier" (larger images).

Embedding:  Compress payload --> base64 encode --> embed data into 2 LSB bits of carrier image

Extracting: Extract 2 LSB bits from carrier until whole image is recieved --> replace the 2 LSB bits w/ random values --> base64 decode extracted data --> decompress --> original image is received

Steganography_tests.py and Steganography_extra_tests.py can be ran to test and view runtimes of various functions
