# ==============================================================================
# 1. SHARED DEFINITIONS (The "Super-Detailed" Visual Lexicon)
# ==============================================================================
SHARED_DEFINITIONS = {
    # --- HUMANS: Role & Uniform Specific (Visual Cues) ---
    "MALE_ROLES":   ["policeman", "firefighter", "soldier", "doctor", "chef", "worker", "mechanic"],
    "FEMALE_ROLES": ["policewoman", "nurse", "waitress", "ballerina", "nun", "bride"],
    "ATHLETES":     ["cyclist", "runner", "skier", "surfer", "player", "swimmer", "skater"],
    "CROWDS":       ["crowd", "audience", "gathering", "mob", "group", "spectators", "crew"],
    
    # --- HUMANS: Age & Family ---
    "MALE_FAMILY":   ["father", "grandfather", "husband", "uncle", "brother", "nephew", "groom"],
    "FEMALE_FAMILY": ["mother", "grandmother", "wife", "aunt", "sister", "niece", "bride"],
    "INFANTS":       ["baby", "newborn", "infant", "toddler"],

    # --- ANIMALS: Visual Sub-Families ---
    "CANINES":      ["bulldog", "beagle", "poodle", "husky", "retriever", "pug", "shepherd", "terrier"],
    "FELINES":      ["tabby", "siamese", "persian", "leopard", "tiger", "lion", "cheetah", "panther"],
    "URSINE":       ["bear", "panda", "polar bear", "grizzly"],
    "EQUINE":       ["horse", "pony", "donkey", "mule", "zebra", "stallion"],
    "BOVINE":       ["cow", "bull", "ox", "bison", "buffalo", "calf"],
    
    # --- BIRDS: Shape Specific ---
    "WATER_FOWL":   ["duck", "goose", "swan", "pelican", "seagull", "penguin"],
    "RAPTORS":      ["eagle", "hawk", "falcon", "owl", "vulture"],
    "SMALL_BIRDS":  ["sparrow", "robin", "pigeon", "parrot", "canary", "hummingbird"],

    # --- VEHICLES: Heavy & Industrial (Bulky visual) ---
    "CONSTRUCTION": ["excavator", "bulldozer", "crane", "forklift", "tractor", "roller"],
    "PUBLIC_TRANS": ["bus", "tram", "trolley", "shuttle", "coach", "double-decker"],
    "FREIGHT":      ["truck", "lorry", "trailer", "tanker", "van"],
    
    # --- VEHICLES: Personal & Sleek ---
    "LUXURY_CARS":  ["limousine", "convertible", "sports car", "sedan", "coupe"],
    "OFFROAD":      ["jeep", "suv", "pickup", "buggy", "quad"],
    "TWO_WHEEL":    ["motorcycle", "scooter", "moped", "bicycle", "bike", "unicycle"],

    # --- WATER & AIR (Distinct Hulls) ---
    "PADDLE_BOAT":  ["canoe", "kayak", "raft", "gondola", "rowboat"],
    "LARGE_SHIP":   ["cruise ship", "ferry", "tanker", "battleship", "cargo ship"],
    "SAIL_BOAT":    ["sailboat", "yacht", "schooner", "catamaran"],
    "ROTOR_CRAFT":  ["helicopter", "drone", "chopper"],
    "WINGED_AIRCRAFT": ["airplane", "jet", "biplane", "glider", "seaplane"],

    # --- NATURE: Geological Textures ---
    "ROCK_FORMATIONS": ["cliff", "boulder", "rock", "crag", "stone", "cave", "arch"],
    "ELEVATION":       ["mountain", "peak", "summit", "ridge", "volcano", "hill", "dune"],
    "WATER_BODIES":    ["ocean", "sea", "lake", "river", "pond", "creek", "stream", "lagoon"],
    "COASTAL":         ["beach", "shore", "coast", "sand", "pier", "harbor", "dock"],

    # --- FLORA: Structural Differences ---
    "CANOPY":       ["tree", "palm", "oak", "pine", "willow", "forest", "woods"],
    "SCRUB":        ["bush", "hedge", "shrub", "thicket", "bramble", "cactus"],
    "FLORAL":       ["rose", "tulip", "sunflower", "daisy", "lily", "orchid", "bouquet"],
    "GROUND_GREEN": ["grass", "lawn", "moss", "clover", "turf", "field", "pasture"],

    # --- ARCHITECTURE: Styles & Parts ---
    "CLASSIC_ARCH": ["castle", "palace", "cathedral", "church", "temple", "ruins", "monument"],
    "MODERN_ARCH":  ["skyscraper", "office", "apartment", "condo", "stadium", "terminal"],
    "RESIDENTIAL":  ["house", "cottage", "cabin", "shack", "bungalow", "villa", "mansion"],
    "ROOFING":      ["roof", "chimney", "dome", "spire", "steeple", "tiles", "shingles"],
    "ENTRANCES":    ["door", "gate", "archway", "entrance", "porch", "balcony", "terrace"],

    # --- INDOOR: Furniture Clusters ---
    "SEATING_SOFT": ["sofa", "couch", "armchair", "recliner", "loveseat", "beanbag"],
    "SEATING_HARD": ["chair", "stool", "bench", "pew", "barstool"],
    "SURFACES":     ["table", "desk", "counter", "island", "vanity", "nightstand"],
    "STORAGE":      ["cabinet", "shelf", "bookcase", "cupboard", "wardrobe", "drawer", "locker"],
    "SLEEPING":     ["bed", "cot", "crib", "hammock", "mattress", "bunk"],

    # --- SMALL OBJECTS: Shape Confusion ---
    "RECTANGLES":   ["book", "tablet", "laptop", "folder", "frame", "painting", "sign"],
    "CYLINDERS":    ["bottle", "can", "jar", "vase", "candle", "column", "pipe", "soda"],
    "HANDHELD":     ["phone", "remote", "calculator", "controller", "wallet", "camera"],
    "TOOLS":        ["hammer", "wrench", "screwdriver", "drill", "saw", "axe", "shovel"],
    
    # --- FOOD: Visual Texture ---
    "ROUND_FRUIT":  ["apple", "orange", "peach", "tomato", "onion", "plum"],
    "ELONGATED":    ["banana", "carrot", "cucumber", "baguette", "sausage", "corn"],
    "SLICES":       ["pizza", "pie", "cake", "quiche", "tart", "cheese"],
    "LIQUIDS":      ["soup", "coffee", "tea", "juice", "wine", "beer", "water"],

    # --- CLOTHING: Material/Cut ---
    "OUTERWEAR":    ["jacket", "coat", "parka", "blazer", "vest", "raincoat", "hoodie"],
    "TOPS":         ["shirt", "t-shirt", "blouse", "sweater", "jersey", "tank top"],
    "BOTTOMS":      ["pants", "jeans", "shorts", "trousers", "skirt", "leggings"],
    "HEADGEAR":     ["hat", "cap", "helmet", "beanie", "crown", "veil", "hood"],
    "FOOTWEAR":     ["shoe", "boot", "sneaker", "sandal", "heel", "slipper", "sock"]
}

# ==============================================================================
# 2. SHARED MAPPING (The Contextual Swap Logic)
# ==============================================================================
# This defines "Hard Negatives". 
# We swap items that appear in similar contexts but look different.
SHARED_RELATIONS = {
    # --- People: Role Swaps (Context: A person standing there) ---
    "MALE_ROLES": "MALE_FAMILY",    # Policeman <-> Father
    "MALE_FAMILY": "MALE_ROLES",
    "ATHLETES": "CROWDS",           # Player <-> Spectator (Crucial for sports)
    "CROWDS": "ATHLETES",
    
    # --- Animals: Shape/Context Swaps ---
    "CANINES": "FELINES",           # Dog <-> Cat
    "FELINES": "CANINES",
    "EQUINE": "BOVINE",             # Horse <-> Cow (Grazing animals)
    "BOVINE": "EQUINE",
    "WATER_FOWL": "PADDLE_BOAT",    # Duck <-> Canoe (Things floating on water)
    "RAPTORS": "WINGED_AIRCRAFT",   # Eagle <-> Airplane (Things in sky)
    
    # --- Vehicles: Function Swaps ---
    "CONSTRUCTION": "FREIGHT",      # Excavator <-> Truck
    "PUBLIC_TRANS": "FREIGHT",      # Bus <-> Truck
    "LUXURY_CARS": "OFFROAD",       # Limo <-> Jeep
    "TWO_WHEEL": "ATHLETES",        # Bike <-> Cyclist (Entity vs Object)
    
    # --- Nature: Texture Swaps ---
    "ROCK_FORMATIONS": "ELEVATION", # Cliff <-> Mountain
    "WATER_BODIES": "GROUND_GREEN", # River <-> Field (Horizontal expanses)
    "GROUND_GREEN": "WATER_BODIES",
    "CANOPY": "SCRUB",              # Tree <-> Bush
    "SCRUB": "CANOPY",
    
    # --- Indoors: Shape/Function Swaps ---
    "SEATING_SOFT": "SLEEPING",     # Sofa <-> Bed
    "SLEEPING": "SEATING_SOFT",
    "SEATING_HARD": "SURFACES",     # Chair <-> Table
    "SURFACES": "SEATING_HARD",
    "STORAGE": "APPLIANCES",        # Cabinet <-> Fridge (Boxy verticals)
    
    # --- Objects: Shape Confusion ---
    "RECTANGLES": "HANDHELD",       # Book <-> Phone
    "HANDHELD": "RECTANGLES",
    "CYLINDERS": "TOOLS",           # Bottle <-> Hammer (Handle shape?)
    
    # --- Food: Shape Swaps ---
    "ROUND_FRUIT": "ELONGATED",     # Apple <-> Banana
    "SLICES": "ROUND_FRUIT",        # Pizza <-> Apple
    
    # --- Clothing ---
    "OUTERWEAR": "TOPS",            # Jacket <-> Shirt
    "TOPS": "OUTERWEAR",
    "HEADGEAR": "FOOTWEAR"          # Hat <-> Shoe (Top vs Bottom mismatch)
}

# ==============================================================================
# 3. STRICT PAIRS (Adjectives, Prepositions, States)
# ==============================================================================
STRICT_PAIRS = {
    # Spatial
    "left": ["right"], "right": ["left"],
    "up": ["down"], "down": ["up"],
    "above": ["below", "underneath"], "below": ["above", "atop"],
    "inside": ["outside"], "outside": ["inside"],
    "foreground": ["background", "distance"], "background": ["foreground"],
    "front": ["back", "behind"], "back": ["front"],
    "near": ["far", "distant"], "far": ["near", "close"],
    
    # States of Being
    "open": ["closed", "shut"], "closed": ["open", "ajar"],
    "wet": ["dry", "parched"], "dry": ["wet", "soaked"],
    "clean": ["dirty", "filthy", "muddy"], "dirty": ["clean", "pristine"],
    "old": ["new", "modern", "brand-new"], "new": ["old", "ancient", "vintage"],
    "empty": ["full", "crowded", "packed"], "full": ["empty", "vacant"],
    "broken": ["fixed", "intact", "whole"], "intact": ["broken", "shattered"],
    
    # Visual Attributes (Adjectives)
    "sunny": ["cloudy", "rainy", "overcast"], "cloudy": ["sunny", "clear"],
    "dark": ["bright", "lit", "luminous"], "bright": ["dark", "dim", "gloomy"],
    "blurry": ["sharp", "focused", "clear"], "sharp": ["blurry", "fuzzy"],
    "colorful": ["monochrome", "dull", "black and white"],
    
    # Actions (Gerunds)
    "standing": ["sitting", "lying", "crouching"], 
    "sitting": ["standing", "lying", "walking"],
    "walking": ["running", "standing", "sprinting"],
    "running": ["walking", "standing", "strolling"],
    "smiling": ["frowning", "crying", "serious"],
    "sleeping": ["awake", "playing", "running"]
}

# ==============================================================================
# 4. CATEGORY POOLS (The "Visual Fallback" Reservoir)
# ==============================================================================
# These words are used when we need a random visual object replacement
# that guarantees a semantic shift without breaking grammar.
CATEGORY_POOLS = {
    "COLORS": [
        "red", "blue", "green", "yellow", "black", "white", "purple", "orange", 
        "pink", "gray", "brown", "beige", "golden", "silver", "teal", "maroon", 
        "navy", "turquoise", "crimson", "ivory"
    ],
    
    "MATERIALS": [
        "wooden", "metallic", "glass", "plastic", "ceramic", "leather", "denim", 
        "cotton", "silk", "concrete", "brick", "stone", "paper", "velvet", "rusty"
    ],
    
    "PATTERNS": [
        "striped", "dotted", "checkered", "plaid", "floral", "spotted", "plain"
    ],
    
    "SHAPES": [
        "round", "square", "rectangular", "triangular", "oval", "curved", "straight", "flat"
    ]
}

# ==============================================================================
# 5. NUMBERS (Expanded)
# ==============================================================================
NUMBERS_DICT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "single": 1, "sole": 1, "pair": 2, "couple": 2, "double": 2, 
    "triple": 3, "trio": 3, "dozen": 12, "half-dozen": 6,
    "few": 3, "several": 5, "many": 10, "numerous": 15, "multiple": 4, 
    "crowd": 20, "group": 5
}