# python-fsk-decoder
Simple script for decoding data from FSK transmitter recorded by RTL-SDR

For more details please read https://mightydevices.com/index.php/2019/08/decoding-fsk-transmission-recorded-by-rtl-sdr-dongle/  
For additional detail about Decode-aq.py, please read: https://iangoegebuer.com/2022/03/decoding-fsk-data-with-an-rtl-sdr-and-python-cheap-30-sdr/  

# Usage

If using just raw audio data:
```
python Decode-aq.py -i data-aq.wav
```

If demodulating an IQ file to use as an AQ file:  
```
python Decode.py -i data-iq.wav -d 300 -o test-aq.wav
python Decode-aq.py -i test-aq.wav
```