import os
from lxml import etree

tree = etree.parse(os.path.join("chantiers", "49_2020", "micmac_0", "MEC-Malt", "NuageImProf_STD-MALT_Etape_6.xml"))
root = tree.getroot()
print(root)




with open(os.path.join("chantiers", "49_2020", "micmac_0", "MEC-Malt", "NuageImProf_STD-MALT_Etape_6.xml")) as f:
    for line in f:
        print(line)