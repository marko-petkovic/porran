import requests
from pymatgen.io.cif import CifParser
from io import StringIO

ZEOLITE_CODES = ['ABW', 'ACO', 'AEI', 'AEL', 'AEN', 'AET', 'AFG', 'AFI', 'AFN', 'AFO',
                 'AFR', 'AFS', 'AFT', 'AFV', 'AFX', 'AFY', 'AHT', 'ANA', 'ANO', 'APC', 
                 'APD', 'AST', 'ASV', 'ATN', 'ATO', 'ATS', 'ATT', 'ATV', 'AVE', 'AVL', 
                 'AWO', 'AWW', 'BCT', 'BEC', 'BIK', 'BOF', 'BOG', 'BOZ', 'BPH', 'BRE', 
                 'BSV', 'CAN', 'CAS', 'CDO', 'CFI', 'CGF', 'CGS', 'CHA', '-CHI', '-CLO', 
                 'CON', 'CSV', 'CZP', 'DAC', 'DDR', 'DFO', 'DFT', 'DOH', 'DON', 'EAB', 
                 'EDI', 'EEI', 'EMT', 'EON', 'EOS', 'EPI', 'ERI', 'ESV', 'ETL', 'ETR', 
                 'ETV', 'EUO', 'EWF', 'EWO', 'EWS', '-EWT', 'EZT', 'FAR', 'FAU', 'FER', 
                 'FRA', 'GIS', 'GIU', 'GME', 'GON', 'GOO', 'HEU', '-HOS', 'IFO', 'IFR', 
                 '-IFT', '-IFU', 'IFW', 'IFY', 'IHW', 'IMF', '-ION', 'IRN', 'IRR', '-IRT', 
                 '-IRY', 'ISV', 'ITE', 'ITG', 'ITH', 'ITR', 'ITT', '-ITV', 'ITW', 'IWR', 
                 'IWS', 'IWV', 'IWW', 'JBW', 'JNT', 'JOZ', 'JRY', 'JSN', 'JSR', 'JST', 
                 'JSW', 'JSY', 'JZO', 'JZT', 'KFI', 'LAU', 'LEV', 'LIO', '-LIT', 'LOS', 
                 'LOV', 'LTA', 'LTF', 'LTJ', 'LTL', 'LTN', 'MAR', 'MAZ', 'MEI', 'MEL', 
                 'MEP', 'MER', 'MFI', 'MFS', 'MON', 'MOR', 'MOZ', 'MRT', 'MSE', 'MSO', 
                 'MTF', 'MTN', 'MTT', 'MTW', 'MVY', 'MWF', 'MWW', 'NAB', 'NAT', 'NES', 
                 'NON', 'NPO', 'NPT', 'NSI', 'OBW', 'OFF', 'OKO', 'OSI', 'OSO', 'OWE', 
                 '-PAR', 'PAU', 'PCR', 'PHI', 'PON', 'POR', 'POS', 'PSI', 'PTF', 'PTO', 
                 'PTT', 'PTY', 'PUN', 'PWN', 'PWO', 'PWW', 'RFE', 'RHO', '-RON', 'RRO', 
                 'RSN', 'RTE', 'RTH', 'RUT', 'RWR', 'RWY', 'SAF', 'SAO', 'SAS', 'SAT', 
                 'SAV', 'SBE', 'SBN', 'SBS', 'SBT', 'SEW', 'SFE', 'SFF', 'SFG', 'SFH', 
                 'SFN', 'SFO', 'SFS', 'SFW', 'SGT', 'SIV', 'SOD', 'SOF', 'SOR', 'SOS', 
                 'SOV', 'SSF', '-SSO', 'SSY', 'STF', 'STI', 'STT', 'STW', '-SVR', 'SVV', 
                 'SWY', '-SYT', 'SZR', 'TER', 'THO', 'TOL', 'TON', 'TSC', 'TUN', 'UEI', 
                 'UFI', 'UOS', 'UOV', 'UOZ', 'USI', 'UTL', 'UWY', 'VET', 'VFI', 'VNI', 
                 'VSV', 'WEI', '-WEN', 'YFI', 'YUG', 'ZON']


def get_zeolite(zeolite_code: str):
    """
    Get the CIF file for a zeolite given its code.
    
    Args:
        zeolite_code (str): The code of the zeolite.
        
    Returns:
        Structure: Zeolite loaded in a pymatgen Structure object.
    """
    if zeolite_code not in ZEOLITE_CODES:
        raise ValueError(f'Zeolite code {zeolite_code} not found.')
    
    url = f'https://europe.iza-structure.org/IZA-SC/cif/{zeolite_code}.cif'
    response = requests.get(url)
    response.raise_for_status()
    cif_string = StringIO(response.text)
    
    parser = CifParser(cif_string, check_cif=False)
    structure = parser.parse_structures(primitive=False)[0]

    return structure