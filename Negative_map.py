# ==============================================================================
# 1. SHARED DEFINITIONS (Rozszerzone o brakujące obiekty)
# ==============================================================================
SHARED_DEFINITIONS = {
    "MALE_HUMAN":   ["man", "boy", "gentleman", "guy", "husband", "father", "son", "male", "person", "persons"],
    "FEMALE_HUMAN": ["woman", "girl", "lady", "wife", "mother", "daughter", "female", "person", "persons"],
    
    "CHILD": ["baby", "toddler", "child", "kid", "boy", "girl"],
    "ADULT": ["adult", "man", "woman", "senior", "elder", "parent"],
    
    "DOG_FAMILY": ["dog", "puppy", "canine", "bulldog", "poodle", "husky"],
    "CAT_FAMILY": ["cat", "kitten", "feline", "tabby"],
    
    "LAND_VEHICLE": ["car", "truck", "bus", "van", "taxi", "vehicle", "bike", "bicycle", "motorcycle"],
    "WATER_VEHICLE": ["boat", "ship", "yacht", "ferry"],
    "AIR_VEHICLE": ["airplane", "plane", "jet", "aircraft", "helicopter", "drone"],
    "RAIL_VEHICLE": ["train", "locomotive", "subway", "tram"],
    
    "NATURE": ["grass", "tree", "trees", "plant", "plants", "bush", "flower", "leaf", "leaves", "hill", "hills", "mountain"],
    "STRUCTURES": ["building", "buildings", "house", "tower", "skyscraper", "fence", "wall", "walls", "ceiling", "floor"],
    
    "STREET_ELEMENTS": ["pole", "poles", "signal", "sign", "board", "traffic light", "traffic lights", "track", "tracks", "road", "street", "sidewalk"],
    "SKY_ELEMENTS": ["sky", "cloud", "clouds", "sun", "moon", "star"],
    
    "HYGIENE": ["washbasin", "sink", "tap", "taps", "toothbrush", "toothpaste", "hanger", "dustbin", "mirror"],
    "CONTAINERS_STORAGE": ["box", "boxed", "container", "bag", "suitcase", "handle", "bin", "basket", "tray"],
    "SPORTS_EQUIP": ["glove", "gloves", "bat", "ball", "racket", "ski", "skis", "stick", "sticks", "helmet"],
    "CLOTHING": ["jacket", "shirt", "pants", "hat", "cap", "shoes", "glasses"],

    # PRZENIESIONE Z POOLS DO DEFINITIONS (Aby uniknąć KeyError):
    "FOOD_ITEMS": ["pizza", "burger", "fruit", "food", "meat", "vegetable", "cake", "bread", "soup"],
    "FURNITURE": ["table", "chair", "sofa", "bed", "desk", "cabinet", "shelf", "bench", "couch"],
    
    "VEGETABLES": ["broccoli", "carrot", "onion", "tomato", "cucumber", "cauliflower", "veggies", "capsicum", "zucchini", "garlic"],
    "FRUITS": ["apple", "banana", "fruit", "strawberry", "orange", "grape", "lemon"],
    
    "KITCHEN_UTENSILS": ["fork", "knife", "spoon", "utensil", "utensils", "scissors", "cutter", "ladle"],
    "DINING_WARE": ["plate", "bowl", "cup", "saucer", "glass", "wineglass", "mug", "tray", "bottle", "jar"],
    
    "PERSONAL_ITEMS": ["wallet", "keys", "passport", "boarding pass", "papers", "id card", "watch", "handbag", "bag", "comb"],
    "OFFICE_ELECTRONICS": ["laptop", "computer", "monitor", "keyboard", "mouse", "printer", "speaker", "tablet"],
    "MEDIA_DEVICES": ["tv", "television", "remote", "phone", "mobile", "ipod", "headset", "camera"],
    
    "BEDDING_DECOR": ["pillow", "pillows", "blanket", "curtain", "curtains", "blinds", "mat", "carpet", "towel"],
    "STATIONERY": ["book", "books", "pen", "pencil", "notebook", "paper", "pamphlet"]
}

# ==============================================================================
# 2. SHARED MAPPING (Relacje wymuszające zmianę sensu)
# ==============================================================================
SHARED_RELATIONS = {
    "MALE_HUMAN": "FEMALE_HUMAN",
    "FEMALE_HUMAN": "MALE_HUMAN",
    "CHILD": "ADULT",
    "ADULT": "CHILD",
    "DOG_FAMILY": "CAT_FAMILY",
    "CAT_FAMILY": "DOG_FAMILY",
    "LAND_VEHICLE": "RAIL_VEHICLE",
    "RAIL_VEHICLE": "AIR_VEHICLE",
    "AIR_VEHICLE": "WATER_VEHICLE",
    "WATER_VEHICLE": "LAND_VEHICLE",
    "NATURE": "STRUCTURES",
    "STRUCTURES": "NATURE",
    "STREET_ELEMENTS": "SKY_ELEMENTS",
    "SKY_ELEMENTS": "STREET_ELEMENTS",
    "HYGIENE": "FOOD_ITEMS",       # Teraz FOOD_ITEMS jest w DEFINITIONS - OK
    "SPORTS_EQUIP": "FURNITURE",   # Teraz FURNITURE jest w DEFINITIONS - OK
    "CLOTHING": "NATURE",
    "VEGETABLES": "FRUITS",
    "FRUITS": "VEGETABLES",
    "KITCHEN_UTENSILS": "STATIONERY",
    "STATIONERY": "KITCHEN_UTENSILS",
    "DINING_WARE": "CONTAINERS_STORAGE",
    "OFFICE_ELECTRONICS": "MEDIA_DEVICES",
    "MEDIA_DEVICES": "OFFICE_ELECTRONICS",
    "PERSONAL_ITEMS": "CLOTHING",
    "BEDDING_DECOR": "STRUCTURES"
}

# ==============================================================================
# 3. STRICT PAIRS (Wzbogacone o czas i relacje)
# ==============================================================================
STRICT_PAIRS = {
    "left": ["right"], "right": ["left"],
    "up": ["down"], "down": ["up"],
    "top": ["bottom"], "bottom": ["top"],
    "inside": ["outside"], "outside": ["inside"],
    "foreground": ["background"], "background": ["foreground"],
    "front": ["back"], "back": ["front"],
    "day": ["night"], "night": ["day"],
    "center": ["edge", "side"], "middle": ["edge", "side"],
    "open": ["closed"], "closed": ["open"],
    "wet": ["dry"], "dry": ["wet"],
    "above": ["below"], "below": ["above"],
    "atop": ["under", "below"], "under": ["above", "atop"],
    "indoor": ["outdoor"], "outdoor": ["indoor"],
    "indoors": ["outdoors"], "outdoors": ["indoors"],
    "cluttered": ["empty", "neat"], "empty": ["full", "cluttered"],
    "full": ["empty"],
    "visible": ["hidden", "blocked"], "hidden": ["visible"]
}

# ==============================================================================
# 4. CATEGORY POOLS (Szybkie mieszanie obiektów)
# ==============================================================================
CATEGORY_POOLS = {
    "COLORS": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "gray", "brown"],
    "ROOMS": ["kitchen", "bathroom", "bedroom", "living room", "office", "garage", "hall", "washroom"],
    "SURFACES": ["ground", "floor", "surface", "tabletop", "snow", "dirt", "sand"],
    "ANIMALS": ["dog", "cat", "bird", "eagle", "bear", "robot", "teddy", "animal"],
    "MATERIALS": ["wooden", "glass", "metal", "plastic", "ceramic", "cloth", "cardboard"],
    "ELECTRONICS_ALL": ["laptop", "phone", "tv", "mouse", "keyboard", "monitor", "remote", "camera"]
}

# --- 5. LICZBY ---
NUMBERS_DICT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
    "few": 3, "many": 10, "several": 5, "numerous": 15
}