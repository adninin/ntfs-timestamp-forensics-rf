# ==========================================
# MAPPING 12 SKENARIO TIMESTAMP MANIPULATION
# Berdasarkan Oh et al. (2024)
# ==========================================

scenario_mapping = {
    1: {
        "tool": "NewFileTime",
        "method": "SetFileTime() API",
        "target": "$SI-C",
        "target_description": "Manipulasi Creation Time"
    },

    2: {
        "tool": "NewFileTime",
        "method": "SetFileTime() API",
        "target": "$SI-M",
        "target_description": "Manipulasi Modified Time"
    },

    3: {
        "tool": "NewFileTime",
        "method": "SetFileTime() API",
        "target": "$SI-MAC",
        "target_description": "Manipulasi Modified, Accessed, Creation"
    },

    4: {
        "tool": "PowerShell",
        "method": "Get-Item Cmdlet",
        "target": "$SI-C",
        "target_description": "Manipulasi Creation Time"
    },

    5: {
        "tool": "PowerShell",
        "method": "Get-Item Cmdlet",
        "target": "$SI-M",
        "target_description": "Manipulasi Modified Time"
    },

    6: {
        "tool": "PowerShell",
        "method": "Get-Item Cmdlet",
        "target": "$SI-MAC",
        "target_description": "Manipulasi Modified, Accessed, Creation"
    },

    7: {
        "tool": "nTimestomp",
        "method": "NtSetInformationFile() API",
        "target": "$SI-C",
        "target_description": "Manipulasi Creation Time"
    },

    8: {
        "tool": "nTimestomp",
        "method": "NtSetInformationFile() API",
        "target": "$SI-M",
        "target_description": "Manipulasi Modified Time"
    },

    9: {
        "tool": "nTimestomp",
        "method": "NtSetInformationFile() API",
        "target": "$SI-MACE",
        "target_description": "Manipulasi seluruh timestamp $SI"
    },

    10: {
        "tool": "File System Tunneling",
        "method": "File System Tunneling",
        "target": "$SI-C",
        "target_description": "Perubahan Creation Time melalui tunneling"
    },

    11: {
        "tool": "SetMACE",
        "method": "NtSetInformationFile() API",
        "target": "$SI-MACE",
        "target_description": "Menggunakan timestamp file lain"
    },

    12: {
        "tool": "SetMACE",
        "method": "NtSetInformationFile() API",
        "target": "$SI-MACE + $FN-MACE",
        "target_description": "Manipulasi timestamp $SI dan $FN"
    }
}


# ==========================================
# MENAMPILKAN MAPPING
# ==========================================

for scenario, info in scenario_mapping.items():

    print(f"Scenario {scenario}")
    print(f"  Tool       : {info['tool']}")
    print(f"  Method     : {info['method']}")
    print(f"  Target     : {info['target']}")
    print(f"  Description: {info['target_description']}")
    print("-" * 50)