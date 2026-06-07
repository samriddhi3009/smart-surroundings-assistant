
import spacy
from ultralytics import YOLO
from PIL import Image

nlp = spacy.load("en_core_web_sm")
model = YOLO("yolov8n.pt")

ignore_verbs = {"use", "can", "could", "would", "should", "may",
                "might", "must", "need", "want", "do", "be", "have",
                "get", "help"}

synonyms = {"slice": "cut", "chop": "cut", "snip": "cut", "scribble": "write"}

def extract_action(question):
    doc = nlp(question)
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.lemma_.lower()
            if verb in synonyms:
                verb = synonyms[verb]
            if verb not in ignore_verbs:
                return verb
    return None

 action_dict = {
    # ==========================================
    # 1. KITCHEN, FOOD & DINING (expanded)
    # ==========================================
    "chop":        ["knife", "cleaver", "chopping board", "blade", "meat cleaver", "hatchet", "kitchen shears"],
    "sit":         ["chair" , "table" , "sofa" ,"bed" ,"bench" , "stool" , "floor"],
    "slice":       ["knife", "mandoline", "slicer", "blade", "chef's knife", "electric knife", "serrated knife"],
    "dice":        ["knife", "chopping board", "dicers", "vegetable chopper"],
    "mince":       ["knife", "meat grinder", "food processor", "mincing knife", "garlic press"],
    "peel":        ["peeler", "knife", "vegetable peeler", "paring knife", "serrated peeler", "steamer for peeling"],
    "grate":       ["grater", "food processor", "microplane", "zester", "cheese grater", "nutmeg grater"],
    "whisk":       ["whisk", "electric mixer", "fork", "frother", "balloon whisk"],
    "knead":       ["dough hook", "hands", "stand mixer", "bread machine", "rolling pin (for folding)"],
    "roll":        ["rolling pin", "pasta roller", "dough roller", "bottle", "cylinder"],
    "spread":      ["spatula", "butter knife", "palette knife", "spreader", "brush"],
    "marinate":    ["bowl", "ziplock bag", "container", "brush", "syringe (for injection)"],
    "season":      ["salt shaker", "pepper mill", "spoon", "hands", "seasoning grinder"],
    "blend":       ["blender", "immersion blender", "food processor", "mixer", "smoothie maker"],
    "strain":      ["colander", "sieve", "cheesecloth", "strainer", "chinois", "filter"],
    "skim":        ["skimmer", "ladle", "spoon", "fat separator"],
    "tenderize":   ["meat mallet", "tenderizing hammer", "fork", "marinade injector"],
    "glaze":       ["brush", "spatula", "spoon", "pastry brush"],
    "steam":       ["steamer basket", "steamer pot", "bamboo steamer", "rice cooker", "pressure cooker (steam mode)"],
    "roast":       ["oven", "roasting pan", "rotisserie", "spit", "air fryer"],
    "grill":       ["grill", "grill pan", "barbecue tongs", "gridiron", "charcoal grill"],
    "saute":       ["saute pan", "skillet", "spatula", "wok", "wooden spoon"],
    "simmer":      ["pot", "stove", "slow cooker", "casserole dish"],
    "caramelize":  ["pan", "stove", "torch (for sugar)", "blow torch"],
    "broil":       ["oven (broil setting)", "broiler pan", "toaster oven"],
    "poach":       ["poaching pan", "pot", "egg poacher", "thermometer"],
    "baste":       ["baster", "brush", "ladle", "spoon", "turkey baster"],
    "deglaze":     ["pan", "wooden spoon", "wine or liquid", "spatula"],
    "ferment":     ["fermentation crock", "jar", "airlock", "weight", "pickling jar"],
    "can":         ["canning jar", "pressure canner", "water bath canner", "lid lifter", "funnel"],
    "freeze":      ["freezer bag", "freezer container", "ice cube tray", "vacuum sealer", "freezer paper"],
    "thaw":        ["refrigerator", "microwave (defrost)", "cold water bowl", "thawing tray"],
    "reheat":      ["microwave", "oven", "stove", "air fryer", "toaster oven", "steamer"],
    "serve":       ["plate", "bowl", "tray", "serving spoon", "ladle", "tongs", "platter"],
    "plate":       ["plate", "tweezers (for plating)", "ring mold", "squeeze bottle", "spatula"],
    "garnish":     ["tweezers", "knife", "peeler", "herb stripper", "microplane"],
    "taste":       ["spoon", "fork", "finger", "tasting spoon", "straw"],
    "smell":       ["nose", "hand (wafting)", "sniffer", "aroma cup"],

    # ==========================================
    # 2. BAKING & PASTRY (new category)
    # ==========================================
    "sift":        ["sifter", "sieve", "flour sifter", "fine-mesh strainer"],
    "cream":       ["mixer", "spatula", "hand mixer", "stand mixer", "whisk"],
    "fold":        ["spatula", "rubber spatula", "wooden spoon", "whisk (gentle)"],
    "pipe":        ["piping bag", "pastry bag", "nozzle", "tip", "coupler"],
    "dust":        ["dredger", "sifter", "fine sieve", "powder duster"],
    "score":       ["lame", "knife", "scoring blade", "razor blade", "scissors (for dough)"],
    "proof":       ["proofing basket", "bowl", "dough proofer", "warm oven", "plastic wrap"],
    "temper":      ["thermometer", "double boiler", "marble slab", "spatula", "spoon"],
    "glaze":       ["pastry brush", "spatula", "spoon", "piping tip (for fine lines)"],
    "crumb coat":  ["offset spatula", "turntable", "bench scraper", "palette knife"],

    # ==========================================
    # 3. OFFICE, WRITING & CRAFTS (expanded)
    # ==========================================
    "write":       ["pen", "pencil", "marker", "chalk", "stylus", "keyboard", "fountain pen", "gel pen", "quill"],
    "draw":        ["pencil", "pen", "marker", "brush", "crayon", "charcoal", "pastel", "tablet stylus", "chalk"],
    "sketch":      ["sketchbook", "pencil", "charcoal", "eraser", "blending stump", "kneaded eraser"],
    "color":       ["colored pencil", "crayon", "marker", "paint", "pastel", "watercolor brush"],
    "paint":       ["brush", "palette", "canvas", "acrylic", "oil paint", "watercolor", "spray gun", "roller"],
    "erase":       ["eraser", "rubber", "correction fluid", "correction tape", "whiteboard eraser", "kneaded eraser"],
    "cut":         ["scissors", "knife", "cutter", "guillotine", "paper trimmer", "exacto knife"],
    "fold":        ["hands", "bone folder", "ruler", "paper folder", "crease tool"],
    "punch":       ["hole punch", "paper punch", "single hole punch", "adjustable punch"],
    "laminate":    ["laminator", "laminating pouch", "cold laminator", "roll laminator"],
    "bind":        ["binder", "rings", "comb binding machine", "spiral coil", "binder clip", "stapler"],
    "staple":      ["stapler", "staple remover", "heavy-duty stapler", "tacker", "electric stapler"],
    "clip":        ["paperclip", "binder clip", "bulldog clip", "foldback clip", "clothespin"],
    "stamp":       ["rubber stamp", "ink pad", "date stamp", "self-inking stamp", "embosser"],
    "engrave":     ["engraving tool", "stylus", "laser engraver", "burin", "electric engraver"],
    "label":       ["label maker", "sticker", "tape", "handwritten label", "printer label"],
    "highlight":   ["highlighter", "marker", "gel highlighter", "fluorescent pen", "digital highlighter"],
    "sign":        ["pen", "stylus", "seal", "signature stamp", "digital signature pad"],
    "notarize":    ["notary seal", "stamp", "embosser", "journal", "pen"],
    "shred":       ["paper shredder", "scissors", "strip-cut shredder", "cross-cut shredder"],
    "file":        ["file folder", "hanging folder", "cabinet", "label", "tab divider", "archival box"],
    "organize":    ["binder", "folder", "tray", "sorter", "label maker", "divider", "storage box"],

    # ==========================================
    # 4. WORKSHOP, TOOLS & CONSTRUCTION (expanded)
    # ==========================================
    "hammer":      ["hammer", "mallet", "sledgehammer", "ball-peen hammer", "claw hammer", "rubber mallet"],
    "screw":       ["screwdriver", "drill", "hex key", "allen wrench", "power screwdriver", "impact driver"],
    "nail":        ["hammer", "nail gun", "nail", "brad nailer", "finish nailer", "pneumatic nailer"],
    "drill":       ["drill", "drill bit", "hammer drill", "power drill", "hand drill", "drill press"],
    "unscrew":     ["screwdriver", "impact driver", "power drill (reverse)", "wrench", "pliers"],
    "tighten":     ["wrench", "spanner", "pliers", "screwdriver", "torque wrench", "socket wrench", "vise grip"],
    "loosen":      ["wrench", "pliers", "penetrating oil", "hammer (tap)", "breaker bar"],
    "measure":     ["ruler", "tape measure", "scale", "protractor", "caliper", "yardstick", "laser measure"],
    "level":       ["spirit level", "laser level", "water level", "torpedo level", "digital level"],
    "clamp":       ["clamp", "vise", "c-clamp", "bar clamp", "spring clamp", "quick-grip clamp"],
    "sand":        ["sandpaper", "orbital sander", "file", "rasp", "sandpaper block", "belt sander", "sponge sander"],
    "file":        ["file", "rasp", "nail file", "metal file", "diamond file", "needle file"],
    "saw":         ["handsaw", "hacksaw", "chainsaw", "jigsaw", "circular saw", "table saw", "bandsaw", "miter saw"],
    "cut":         ["cutter", "scissors", "knife", "plasma cutter", "shears", "tin snips", "bolt cutter"],
    "weld":        ["welding torch", "arc welder", "mask", "soldering iron", "MIG welder", "TIG welder", "welding helmet"],
    "solder":      ["soldering iron", "solder wire", "flux", "desoldering pump", "soldering station"],
    "glue":        ["glue gun", "wood glue", "super glue", "epoxy", "spread glue", "brush", "applicator"],
    "pry":         ["crowbar", "pry bar", "nail puller", "wrecking bar", "hammer (claw)"],
    "scrape":      ["scraper", "putty knife", "razor blade", "paint scraper", "glass scraper"],
    "paint":       ["paintbrush", "roller", "spray gun", "paint tray", "tape (masking)", "drop cloth"],
    "varnish":     ["varnish brush", "rag", "sprayer", "foam brush", "pad applicator"],
    "polish":      ["polishing cloth", "buffer", "polishing machine", "compound", "wax applicator"],
    "grind":       ["angle grinder", "bench grinder", "grinding wheel", "dremel", "abrasive disc"],
    "chisel":      ["chisel", "mallet", "wood chisel", "cold chisel", "firmer chisel"],
    "carve":       ["carving knife", "gouge", "chisel", "dremel", "wood carving tool", "burin"],
    "plane":       ["hand plane", "electric planer", "jointer", "thickness planer", "block plane"],
    "route":       ["router", "router bit", "trimmer", "laminate router", "plunge router"],

    # ==========================================
    # 5. CLEANING & HOUSEHOLD MAINTENANCE (expanded)
    # ==========================================
    "clean":       ["cloth", "mop", "broom", "sponge", "brush", "vacuum cleaner", "steam cleaner", "pressure washer"],
    "sweep":       ["broom", "dustpan", "brush", "push broom", "carpet sweeper", "street broom"],
    "mop":         ["mop", "bucket", "spray mop", "spin mop", "string mop", "flat mop"],
    "wash":        ["soap", "sponge", "brush", "cloth", "detergent", "washing machine", "dishwasher", "power washer"],
    "wipe":        ["cloth", "tissue", "sponge", "mop", "squeegee", "paper towel", "microfiber cloth", "wet wipe"],
    "scrub":       ["scrubbing brush", "scouring pad", "steel wool", "scrub brush", "sponge with scrubber", "brush"],
    "vacuum":      ["vacuum cleaner", "handvac", "robot vacuum", "stick vacuum", "canister vacuum", "wet-dry vac"],
    "dust":        ["duster", "microfiber cloth", "feather duster", "electrostatic duster", "dusting spray"],
    "polish":      ["polishing cloth", "wax", "furniture polish", "buffer", "spray polish"],
    "degrease":    ["degreaser spray", "sponge", "brush", "cloth", "pressure washer", "steam cleaner"],
    "sanitize":    ["sanitizer spray", "wipe", "UV light", "steam cleaner", "disinfectant fogger"],
    "disinfect":   ["bleach solution", "alcohol wipe", "UV sterilizer", "spray bottle", "disinfectant wipe"],
    "unclog":      ["plunger", "drain snake", "chemical cleaner", "drain auger", "plumber's snake", "baking soda + vinegar"],
    "organize":    ["bin", "shelf", "label maker", "drawer divider", "storage box", "hanger", "basket"],

    # ==========================================
    # 6. LAUNDRY & FABRIC CARE (new)
    # ==========================================
    "wash":        ["washing machine", "detergent", "fabric softener", "hand wash basin", "soap bar"],
    "dry":         ["dryer", "clothesline", "drying rack", "tumble dryer", "spin dryer", "hair dryer (small items)"],
    "iron":        ["iron", "ironing board", "steamer", "steam iron", "travel iron", "garment steamer"],
    "fold":        ["hands", "folding board", "folding table"],
    "hang":        ["hanger", "clothespin", "clothesline", "rack", "wooden hanger", "padded hanger"],
    "steam":       ["garment steamer", "steam iron", "steam closet", "hand steamer"],
    "mend":        ["needle", "thread", "thimble", "sewing kit", "darning mushroom", "patch"],
    "starch":      ["starch spray", "iron", "spray bottle", "starch powder + water"],
    "bleach":      ["bleach", "bleach pen", "spray bleach", "bucket", "gloves"],
    "stain remove": ["stain remover spray", "bar soap", "toothbrush (for scrubbing)", "bleach pen", "baking soda paste"],
    "dry clean":   ["dry cleaning machine", "solvent", "dry cleaning bag (home)", "professional service"],

    # ==========================================
    # 7. GARDENING & OUTDOORS (expanded)
    # ==========================================
    "dig":         ["shovel", "spade", "trowel", "mattock", "excavator", "post hole digger", "spading fork"],
    "water":       ["watering can", "hose", "sprinkler", "drip irrigation system", "water wand", "garden sprayer"],
    "prune":       ["shears", "secateurs", "loppers", "pruning saw", "hedge trimmers", "pole pruner"],
    "rake":        ["rake", "leaf blower", "bow rake", "leaf rake", "thatch rake", "lawn rake"],
    "mow":         ["lawnmower", "weed whacker", "scythe", "reel mower", "riding mower", "robot mower"],
    "plant":       ["pot", "trowel", "seed tray", "dibber", "transplanter", "garden fork", "planting auger"],
    "weed":        ["hoe", "weed puller", "gloves", "hand fork", "cultivator", "weed torch"],
    "fertilize":   ["fertilizer spreader", "hand sprayer", "granule applicator", "liquid fertilizer injector"],
    "mulch":       ["mulch spreader", "shovel", "rake", "wheelbarrow", "tarp"],
    "sow":         ["seed spreader", "hand seeder", "seed tape", "dibber", "broadcaster"],
    "harvest":     ["harvest knife", "garden shears", "basket", "gloves", "pruning shears", "fruit picker"],
    "graft":       ["grafting knife", "grafting tape", "pruning shears", "grafting wax", "rubber bands"],
    "propagate":   ["cutting tool", "rooting hormone", "small pot", "plastic bag", "humidity dome"],
    "spray":       ["spray bottle", "garden sprayer", "backpack sprayer", "hose-end sprayer", "atomizer"],
    "trim":        ["hedge trimmer", "shears", "scissors", "grass clippers", "weed whacker"],
    "blow":        ["leaf blower", "air blower (handheld)", "backpack blower", "gas blower"],

    # ==========================================
    # 8. AUTOMOTIVE & VEHICLE CARE (expanded)
    # ==========================================
    "lift":        ["jack", "jack stand", "car lift", "hydraulic jack", "scissor jack", "floor jack"],
    "inflate":     ["air pump", "compressor", "pressure gauge", "bike pump", "tire inflator", "foot pump"],
    "jumpstart":   ["jumper cables", "battery pack", "jump starter", "booster cables", "power bank (auto)"],
    "tow":         ["tow rope", "winch", "tow hitch", "tow strap", "recovery rope", "snatch strap"],
    "wash":        ["pressure washer", "sponge", "car shampoo", "bucket", "mitt", "dry cloth"],
    "wax":         ["wax applicator", "buffer", "microfiber cloth", "liquid wax", "paste wax", "spray wax"],
    "scrub":       ["scrub brush", "clay bar", "detailing brush", "toothbrush (small areas)"],
    "polish":      ["polisher", "polishing pad", "compound", "buffer", "microfiber towel"],
    "change oil":  ["oil filter wrench", "drain pan", "funnel", "new oil", "ramp or jack", "gloves"],
    "replace tire":["lug wrench", "jack", "spare tire", "torque wrench", "wheel chock"],
    "charge battery":["battery charger", "jump starter", "trickle charger", "alternator (when driving)"],
    "bleed brakes":["bleeder kit", "wrench", "brake fluid", "clear tube", "bottle", "c-clamp (for caliper)"],
    "align":       ["alignment machine", "string", "laser alignment tool", "tape measure", "turn plates"],

    # ==========================================
    # 9. COMPUTING & DIGITAL (expanded)
    # ==========================================
    "type":        ["keyboard", "keypad", "mechanical keyboard", "virtual keyboard", "ergonomic keyboard", "stenotype"],
    "click":       ["mouse", "trackpad", "stylus", "touchscreen", "pen", "voice command"],
    "navigate":    ["mouse", "trackpad", "touchscreen", "joystick", "trackball", "stylus", "keyboard arrows"],
    "scroll":      ["mouse wheel", "trackpad (two-finger)", "touchscreen", "arrow keys", "scroll wheel"],
    "store":       ["hard drive", "flash drive", "SD card", "cloud storage", "SSD", "external drive", "NAS", "tape drive"],
    "backup":      ["external hard drive", "cloud service", "backup software", "NAS", "tape backup", "USB stick"],
    "print":       ["printer", "ink cartridge", "paper", "3D printer", "laser printer", "inkjet printer", "thermal printer"],
    "scan":        ["scanner", "copier", "all-in-one printer", "drum scanner", "film scanner", "document feeder"],
    "copy":        ["copier", "all-in-one printer", "scanner + printer", "duplicator"],
    "fax":         ["fax machine", "online fax service", "all-in-one printer with fax", "phone line"],
    "encrypt":     ["encryption software", "USB dongle", "smart card", "TPM chip", "YubiKey", "password manager"],
    "charge":      ["charger", "cable", "adapter", "power bank", "docking station", "wireless charger", "battery charger"],
    "boot":        ["power button", "BIOS key", "bootable USB", "hard drive", "SSD", "boot manager"],
    "shut down":   ["power button", "OS shutdown menu", "remote command", "smart plug", "power strip"],
    "reboot":      ["power button (long press)", "OS restart", "remote command", "reset button"],
    "format":      ["formatting tool (OS)", "disk utility", "command line", "USB flasher", "SD card formatter"],
    "install":     ["installer USB", "CD/DVD", "package manager", "setup wizard", "app store", "script"],

    # ==========================================
    # 10. ELECTRONICS, POWER & LIGHT (expanded)
    # ==========================================
    "charge":      ["charger", "cable", "adapter", "power bank", "docking station", "wireless charger", "battery charger"],
    "light":       ["lighter", "match", "candle", "torch", "lamp", "flashlight", "headlamp", "lantern", "oil lamp"],
    "solder":      ["soldering iron", "solder wire", "flux", "desoldering pump", "soldering station", "hot air rework"],
    "desolder":    ["desoldering pump", "desoldering wick", "hot air station", "soldering iron", "tweezers"],
    "multimeter":  ["digital multimeter", "probes", "clamp meter", "voltage tester", "continuity tester"],
    "oscilloscope":["oscilloscope", "probe", "BNC cable", "signal generator", "function generator"],
    "wire":        ["wire stripper", "pliers", "cutter", "crimper", "soldering iron", "heat shrink", "tape"],
    "crimp":       ["crimping tool", "connector", "wire", "pliers", "ratchet crimper"],
    "strip wire":  ["wire stripper", "knife", "scissors", "automatic wire stripper", "laser stripper"],
    "test voltage":["multimeter", "test light", "voltage pen", "non-contact voltage tester"],
    "splice":      ["wire stripper", "electrical tape", "wire nut", "soldering iron", "heat shrink tubing", "crimp connector"],
    "ground":      ["grounding wire", "ground clamp", "ground rod", "multimeter (continuity)", "ESD strap"],

    # ==========================================
    # 11. MEDICAL, HEALTH & FIRST AID (expanded)
    # ==========================================
    "inject":      ["syringe", "needle", "auto-injector", "pen injector", "prefilled syringe", "injection pen"],
    "bandage":     ["bandage", "plaster", "gauze", "tape", "tourniquet", "elastic bandage", "adhesive bandage", "butterfly closure"],
    "diagnose":    ["stethoscope", "thermometer", "blood pressure cuff", "otoscope", "glucometer", "pulse oximeter", "reflex hammer"],
    "operate":     ["scalpel", "forceps", "clamp", "laser", "surgical scissors", "retractor", "cautery pen", "bone saw"],
    "suture":      ["needle driver", "suture needle", "thread", "forceps", "scissors", "suture kit"],
    "cauterize":   ["cautery pen", "electrocautery machine", "laser", "heated tool", "chemical cautery stick"],
    "splint":      ["splint", "bandage", "tape", "vacuum splint", "SAM splint", "cardboard (improvised)"],
    "immobilize":  ["neck collar", "backboard", "splint", "vacuum mattress", "tape", "bandage"],
    "resuscitate": ["AED", "CPR mask", "bag valve mask", "defibrillator", "mouth barrier"],
    "ventilate":   ["bag valve mask", "ventilator", "respirator", "CPR mask", "oxygen mask", "tube"],
    "monitor":     ["heart monitor", "pulse oximeter", "blood pressure monitor", "ECG machine", "temperature probe"],
    "draw blood":  ["vacutainer", "needle", "tourniquet", "alcohol wipe", "blood collection tube", "butterfly needle"],
    "transfuse":   ["IV bag", "IV line", "needle", "blood filter", "infusion pump", "roller clamp"],
    "measure BP":  ["sphygmomanometer", "stethoscope", "digital BP cuff", "wrist monitor"],
    "check pulse": ["fingers (palpation)", "stethoscope", "pulse oximeter", "heart monitor"],
    "check temp":  ["thermometer", "infrared thermometer", "tympanic thermometer", "mercury thermometer"],
    "irrigate":    ["saline syringe", "bulb syringe", "irrigation bottle", "IV line", "wound irrigation kit"],
    "suction":     ["suction pump", "bulb syringe", "vacuum aspirator", "oral suction tip", "tonsil suction"],

    # ==========================================
    # 12. SPORTS, ATHLETICS & LEISURE (expanded)
    # ==========================================
    "hit":         ["bat", "racket", "club", "stick", "paddle", "mallet", "cricket bat", "table tennis paddle", "fist"],
    "catch":       ["glove", "mitt", "net", "baseball glove", "cricket glove", "fishing net", "bare hands"],
    "throw":       ["hand", "glove", "catapult", "launcher", "baseball", "ball", "frisbee", "javelin"],
    "kick":        ["foot", "boot", "shoe", "shin guard (for opponent contact)", "ball"],
    "jump":        ["rope (skipping)", "box (plyo)", "shoes", "trampoline", "springboard"],
    "run":         ["shoes", "treadmill", "track", "grass", "trail", "barefoot"],
    "swim":        ["goggles", "swim cap", "fins", "kickboard", "pull buoy", "snorkel", "wetsuit"],
    "climb":       ["rope", "harness", "climbing shoes", "chalk bag", "carabiner", "belay device", "helmet"],
    "lift (weights)":["barbell", "dumbbell", "kettlebell", "weight machine", "smith machine", "cable pulley"],
    "squat":       ["barbell", "squat rack", "safety bars", "belt", "knee sleeves", "box (box squat)"],
    "bench press": ["barbell", "bench", "spotter", "weight plates", "rack", "dumbbells"],
    "deadlift":    ["barbell", "weight plates", "lifting straps", "belt", "chalk", "trap bar"],
    "row":         ["rowing machine", "boat", "oar", "ergometer", "rower", "oarlock"],
    "cycle":       ["bicycle", "stationary bike", "spin bike", "pedals", "helmet", "clip-in shoes"],
    "stretch":     ["mat", "yoga strap", "foam roller", "stretching machine", "wall (for wall stretch)"],
    "balance":     ["balance board", "wobble cushion", "bosu ball", "tightrope", "slackline"],
    "throw":       ["ball", "javelin", "discus", "shot put", "hammer (track & field)", "frisbee"],
    "shoot (basketball)":["basketball", "hoop", "backboard", "net", "shooting gun (training aid)"],
    "shoot (archery)":["bow", "arrow", "quiver", "sight", "stabilizer", "release aid", "target"],
    "shoot (firearm)":["rifle", "pistol", "shotgun", "ammunition", "target", "hearing protection", "safety glasses"],

    # ==========================================
    # 13. MUSIC & AUDIO (new)
    # ==========================================
    "play":        ["instrument", "hands", "mouth", "bow", "pick", "drumstick", "mallet", "piano keys", "guitar fret"],
    "strum":       ["guitar", "ukulele", "pick", "finger", "plectrum", "mandolin"],
    "pluck":       ["finger", "pick", "plectrum", "thumb pick", "banjo", "harp", "bass", "guitar"],
    "bow":         ["violin bow", "cello bow", "rosined bow", "bass bow", "horsehair bow"],
    "drum":        ["drumstick", "mallet", "brush", "hands", "feet (bass drum pedal)", "drum", "cajon"],
    "blow":        ["mouth", "reed", "brass mouthpiece", "woodwind instrument", "recorder", "flute", "harmonica"],
    "sing":        ["voice", "microphone", "karaoke machine", "auto-tune (digital)", "vocoder", "headset mic"],
    "record":      ["microphone", "audio interface", "recorder", "DAW (software)", "tape machine", "field recorder"],
    "mix (audio)": ["mixing console", "fader", "knob", "DAW", "EQ", "compressor", "audio interface", "headphones"],
    "master":      ["mastering limiter", "EQ", "monitor speakers", "audio editor", "outboard gear"],
    "amplify":     ["amplifier", "speaker", "PA system", "megaphone", "guitar amp", "power amp"],
    "tune":        ["tuner", "pitch pipe", "tuning fork", "guitar tuner (clip-on)", "app", "ear"],
    "conduct":     ["baton", "hands", "sheet music stand", "orchestra", "conductor's podium"],
    "applaud":     ["hands", "clapper", "applause machine (sound effect)"],

    # ==========================================
    # 14. SCIENCE & LABORATORY (new)
    # ==========================================
    "pipette":     ["pipette", "pipette filler", "micropipette", "multichannel pipette", "electronic pipette"],
    "centrifuge":  ["centrifuge", "rotor", "tubes", "adapters", "microcentrifuge"],
    "mix (lab)":   ["vortex mixer", "magnetic stirrer", "shaker", "rotator", "rocking platform", "stir bar"],
    "heat (lab)":  ["hot plate", "Bunsen burner", "heating mantle", "water bath", "oven", "microwave (lab)"],
    "cool":        ["refrigerator", "freezer", "cryo freezer", "ice bath", "cold plate", "Peltier cooler"],
    "weigh (lab)": ["analytical balance", "precision scale", "microbalance", "weighing boat", "calibration weight"],
    "measure volume":["graduated cylinder", "volumetric flask", "burette", "pipette", "measuring cup"],
    "titrate":     ["burette", "Erlenmeyer flask", "pipette", "indicator", "magnetic stirrer", "stand and clamp"],
    "filter":      ["filter paper", "funnel", "Buchner funnel", "vacuum flask", "filter membrane", "syringe filter"],
    "distill":     ["distillation flask", "condenser", "heat source", "receiving flask", "thermometer", "adapters"],
    "extract":     ["separatory funnel", "solvent", "rotary"]
}
def lookup_verb(question):
    verb = extract_action(question)
    if verb is None:
        return None, []
    candidates = action_dict.get(verb, [])
    return verb, candidates

def find_objects_in_image(image, object_list, confidence_threshold=0.25):
    results = model(image, conf=confidence_threshold)
    detected_objects = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            object_name = model.names[int(box.cls[0].item())]
            confidence = box.conf[0].item()
            if object_name in object_list:
                detected_objects.append(object_name)
                print(f"    Found {object_name} ({confidence:.0%} confidence)")
    return list(set(detected_objects))

def build_answer(verb, found_objects, candidates):
    if verb is None or not candidates:
        return "Sorry, I don't know what objects are needed for that action."
    if not found_objects:
        items = ", ".join(candidates)
        return f"To {verb}, you would need: {items}. But I couldn't find any in the image."
    if len(found_objects) == 1:
        return f"Yes! You can use the {found_objects[0]} to {verb}. I found it in the image."
    if len(found_objects) > 1:
        items = " and ".join(found_objects)
        return f"You can use the {items} to {verb}. I found them in the image."

def full_pipeline(image, question):
    print(f"\n{'='*50}")
    print(f"Question : {question}")
    verb, candidates = lookup_verb(question)
    print(f"Verb     : {verb}")
    print(f"Looking for: {candidates}")
    print("-" * 50)
    if candidates:
        found_objects = find_objects_in_image(image, candidates)
    else:
        found_objects = []
    answer = build_answer(verb, found_objects, candidates)
    print(f"\n💬 Answer: {answer}")
    print("="*50)
    return answer

# ── Run ───────────────────────────────────────────────────────
from google.colab import files
print("📸 Upload an image of your surroundings:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = Image.open(image_path)
print(f"✅ Image loaded!\n")
print("="*50)
print("Type your question below. Type 'exit' to stop.")
print("="*50)
while True:
    question = input("\n❓ Enter your question: ").strip()
    if question.lower() == "exit":
        print("👋 Goodbye!")
        break
    if question == "":
        print("⚠️ Please type a question.")
        continue
    full_pipeline(image, question)
