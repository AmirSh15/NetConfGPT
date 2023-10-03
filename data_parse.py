import os
import pyang
import xml.etree.ElementTree as ET
import xml.etree as etree

def data_parser(yang_file_path, tmp_dir='/home/amir/NetConfGPT/tmp/'):
    """
    This function parses the YANG file and returns the parsed YANG module
    """
    # check if tmp_dir exists
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        
    # first, convert the YANG file to YIN format
    os.system(f'pyang -f yin {yang_file_path} -o {tmp_dir}/tmp.yin')
    
    # read the YIN file as it is in XML format
    with open(f'{tmp_dir}/tmp.yin', 'r') as f:
        yang_data = f.read()
    
    # parse the YIN file
    parser1 = etree.(encoding="utf-8", recover=True)
    tree = ET.fromstring(yang_data)
    yang_module = tree.find('module')
    
    
if __name__ == "__main__":
    yang_file_path = '/home/amir/NetConfGPT/cm-data-model_subset_23_5/ASN_Encoder_Configuration/aecfil.yang'
    data_parser(yang_file_path)