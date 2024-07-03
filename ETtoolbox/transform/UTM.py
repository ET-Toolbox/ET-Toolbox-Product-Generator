from math import floor

def UTM_proj4_from_latlon(lat: float, lon: float) -> str:
    UTM_zone = (floor((lon + 180) / 6) % 60) + 1
    UTM_proj4 = f"+proj=utm +zone={UTM_zone} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    return UTM_proj4

def UTM_proj4_from_zone(zone: str):
    zone_number = int(zone[:2])

    if zone[2].upper() == "N":
        hemisphere = ""
    elif zone[2].upper() == "S":
        hemisphere = "+south "
    else:
        raise ValueError(f"invalid hemisphere in zone: {zone}")

    UTM_proj4 = f"+proj=utm +zone={zone_number} {hemisphere}+datum=WGS84 +units=m +no_defs"

    return UTM_proj4
