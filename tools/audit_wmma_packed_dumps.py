"""CPU-only audit of every complete physical half2 word and its integer fields."""
from pathlib import Path
import hashlib
import json
import struct
import sys

directory=Path(sys.argv[1])
out=Path(sys.argv[2])
lut=[]
for state in range(65536):
    product=state*0x83dcd12d&0xffffffff
    numerator=(1024+sum(product.to_bytes(4,'little')))*1774-1329*2048
    lut.append(struct.unpack('<H',struct.pack('<e',numerator/2**18))[0])
records=[]
for turn in ('first','confirm'):
    actual=(directory/f'{turn}-exhaustive-actual.bin').read_bytes()
    expected=(directory/f'{turn}-exhaustive-expected.bin').read_bytes()
    assert len(actual)==65536*5*8*4 and actual==expected
    distinct=0;samples=[]
    for index,fields in enumerate(struct.iter_unpack('<8I',actual)):
        state=index%65536;low=lut[state];high=lut[65535-state]
        word=low|(high<<16)
        assert fields==(low,high,low,high,state*0x83dcd12d&0xffffffff,low,word,word)
        for start,packed in ((0,6),(2,7)):
            assert fields[start]==fields[packed]&65535
            assert fields[start+1]==fields[packed]>>16
            distinct+=fields[start]!=fields[start+1]
        if index in (0,1,2,65535):
            samples.append(dict(state=state,fields_hex=[f'{v:08x}' for v in fields]))
    records.append(dict(turn=turn,status='PASS',bytes=len(actual),records=65536*5,
                        whole_words_checked=65536*5*2,half_components_checked=65536*5*4,
                        words_with_distinct_low_high=distinct,samples=samples,
                        actual_sha256=hashlib.sha256(actual).hexdigest(),
                        expected_sha256=hashlib.sha256(expected).hexdigest()))
assert records[0]['actual_sha256']==records[1]['actual_sha256']
out.write_text(json.dumps(dict(status='PASS_PHYSICAL_DUMPS_CPU_AUDIT',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    endian='little',reference='Exact integer mul1 numerator, one RN16 per half',
    campaigns=records),indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(records,indent=2))
