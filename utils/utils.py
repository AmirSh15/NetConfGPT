import os


def remove_descriptions(yang_file:str)->str:
    """
    Remove all the descriptions from the yang file.
    """
    
    # first find the "description" keywords
    inds = []
    for i, line in enumerate(yang_file.split("\n")):
        if "description" in line:
            inds.append(i)
        
    nex_txt = ""
    skip_flag = False
    for i , line in enumerate(yang_file.split("\n")):
        if not skip_flag:
            if i not in inds:
                nex_txt += line + "\n"
            if i in inds and not """.";""" in line:
                skip_flag = True
        else:
            if """.";""" in line:
                skip_flag = False
                
    return nex_txt
    
    
if __name__ == "__main__":
    # read test yang file
    yang_file_path = f"/home/amir/NetConfGPT/examples/ocsset.yang"
    with open(yang_file_path, "r") as f:
        yang_file = f.read() 
    
    # remove the descriptions
    yang_file = remove_descriptions(yang_file)
    
    # save the new yang file
    with open(os.path.join("/home/amir/NetConfGPT/examples/", "ocsset_no_desc.yang"), "w") as f:
        f.write(yang_file)
    